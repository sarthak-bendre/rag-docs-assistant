"""Evaluate retrieval (hit rate@k, MRR) and answer faithfulness (RAGAS) across setups.

QA file: eval/qa_pairs.jsonl, one JSON object per line:
  {"id": "q01", "question": "...", "answer": "reference answer",
   "source": "manual.pdf", "pages": [42]}
Use "source": null for questions whose answer is NOT in the corpus (tests abstention).

Run:
  python -m eval.run_eval                       # retrieval metrics for all setups (no LLM needed)
  python -m eval.run_eval --generate            # + answers, abstention, RAGAS faithfulness
  python -m eval.run_eval --generate --setups hybrid hybrid+rerank
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.config import settings
from src.generate import answer
from src.llm import get_chat_model
from src.retrieve import Hit, Retriever

EVAL_DIR = Path(__file__).resolve().parent
SETUPS = {  # name -> (mode, rerank)
    "vector": ("vector", False),
    "bm25": ("bm25", False),
    "hybrid": ("hybrid", False),
    "hybrid+rerank": ("hybrid", True),
}


def load_qa(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_relevant(h: Hit, qa: dict) -> bool:
    if h.source != qa["source"]:
        return False
    pages = qa.get("pages")
    return not pages or any(h.page_start <= p <= h.page_end for p in pages)


def first_relevant_rank(hits: list[Hit], qa: dict) -> int | None:
    return next((r for r, h in enumerate(hits, 1) if is_relevant(h, qa)), None)


def faithfulness(rows: list[dict]) -> list[float]:
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness

    judge = LangchainLLMWrapper(get_chat_model(settings.judge_provider, settings.judge_model))
    ds = EvaluationDataset.from_list(rows)
    res = evaluate(ds, metrics=[Faithfulness()], llm=judge, show_progress=True)
    return res.to_pandas()["faithfulness"].tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", type=Path, default=EVAL_DIR / "qa_pairs.jsonl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--setups", nargs="+", default=list(SETUPS), choices=list(SETUPS))
    ap.add_argument("--generate", action="store_true", help="also generate answers + RAGAS faithfulness")
    args = ap.parse_args()

    qa = load_qa(args.qa)
    answerable = [q for q in qa if q.get("source")]
    unanswerable = [q for q in qa if not q.get("source")]
    print(f"{len(qa)} questions ({len(answerable)} answerable, {len(unanswerable)} unanswerable)")

    retriever = Retriever()
    llm = get_chat_model() if args.generate else None
    summary, per_q = [], []

    for name in args.setups:
        mode, rerank = SETUPS[name]
        print(f"\n=== {name} ===")
        ranks, latencies, ragas_rows, abstain_ok = [], [], [], []
        for q in qa:
            t0 = time.perf_counter()
            if args.generate:
                ans = answer(q["question"], retriever, llm, k=args.k, mode=mode, rerank=rerank)
                hits = ans.hits
            else:
                hits = retriever.search(q["question"], k=args.k, mode=mode, rerank=rerank)
            latencies.append(time.perf_counter() - t0)

            row = {"setup": name, "id": q["id"], "question": q["question"],
                   "retrieved": "; ".join(f"{h.source} {h.page_label}" for h in hits)}
            if q.get("source"):
                r = first_relevant_rank(hits, q)
                ranks.append(r)
                row["rank"] = r
            if args.generate:
                row["response"] = ans.text
                if q.get("source"):
                    row["abstained"] = ans.not_found
                    if not ans.not_found:
                        ragas_rows.append({"user_input": q["question"], "response": ans.text,
                                           "retrieved_contexts": [h.text for h in hits],
                                           "reference": q.get("answer", "")})
                else:
                    abstain_ok.append(ans.not_found)
                    row["abstained"] = ans.not_found
            per_q.append(row)

        n = len(ranks) or 1
        s = {
            "setup": name,
            f"hit@{args.k}": sum(r is not None for r in ranks) / n,
            "hit@1": sum(r == 1 for r in ranks) / n,
            "MRR": sum(1 / r for r in ranks if r) / n,
            "avg_latency_s": sum(latencies) / len(latencies),
        }
        if args.generate:
            if unanswerable:
                s["abstention_acc"] = sum(abstain_ok) / len(abstain_ok)
            s["false_abstain"] = sum(
                1 for row in per_q if row["setup"] == name and row.get("rank") is not None and row.get("abstained")
            )
            if ragas_rows:
                scores = faithfulness(ragas_rows)
                valid = [x for x in scores if x == x]  # drop NaN (judge parse failures)
                s["faithfulness"] = sum(valid) / len(valid) if valid else float("nan")
                s["faithfulness_n"] = len(valid)
        summary.append(s)
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()})

    out = EVAL_DIR / "results"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    df = pd.DataFrame(summary).round(3)
    df.to_csv(out / f"summary-{stamp}.csv", index=False)
    pd.DataFrame(per_q).to_csv(out / f"per_question-{stamp}.csv", index=False)
    print("\n" + df.to_markdown(index=False))
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
