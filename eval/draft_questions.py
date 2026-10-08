"""Draft candidate QA pairs for HUMAN REVIEW (not a substitute for checking them yourself).

Picks fact-dense chunks spread across all manuals, asks the judge LLM for one specific
question + exact answer + verbatim evidence quote, then pins the answer to the exact page(s)
by locating the quote in the PDF.  Output: eval/qa_pairs.draft.jsonl

  python -m eval.draft_questions --n 45
"""
import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import pymupdf as fitz
from langchain_core.prompts import ChatPromptTemplate

from src.config import settings
from src.llm import get_chat_model

EVAL_DIR = Path(__file__).resolve().parent

PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You write evaluation questions for a maintenance-manual search assistant.
From the passage, write ONE question a maintenance engineer might really ask, whose answer is a
specific fact stated in the passage (a number, limit, interval, procedure step, part, or condition).
The question must make sense without seeing the passage (name the equipment/topic; never say
"according to the passage"). Do NOT ask about: worked examples or one-off calculations
("the example", "the pump described"), named example equipment (e.g. "transformer T1"), variable
definitions, equations, publication years, document history, or OSHA/VPP program administration.
Return only JSON:
{{"question": "...", "answer": "short exact answer", "evidence": "verbatim quote of 8-25 words from the passage that contains the answer"}}
If the passage has no concrete fact worth asking about, return {{"skip": true}}."""),
    ("human", "Source: {source}\n\nPassage:\n{text}"),
])

FACT_RE = re.compile(r"\d+(\.\d+)?\s?(%|°|psi|psig|kv|v\b|volts|hp|rpm|hours|months|years|ft|in\.|inch|mm|ohm|megohm|db|kw|amp|nm|lb)", re.I)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def locate_pages(pdf: Path, evidence: str, p_start: int, p_end: int) -> list[int]:
    """Find which page(s) of the chunk's range contain the evidence quote."""
    words = norm(evidence).split()
    probe = " ".join(words[:6])  # first few words are robust to line-joins/hyphenation
    with fitz.open(pdf) as doc:
        hits = [p for p in range(p_start, p_end + 1) if probe and probe in norm(doc[p - 1].get_text())]
    return hits or list(range(p_start, p_end + 1))


def pick_chunks(chunks: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_src = defaultdict(list)
    for c in chunks:
        if len(c["text"]) > 600 and len(FACT_RE.findall(c["text"])) >= 2:
            by_src[c["source"]].append(c)
    for v in by_src.values():
        rng.shuffle(v)
    picked, i = [], 0
    while len(picked) < n and any(by_src.values()):  # round-robin across manuals
        src = sorted(by_src)[i % len(by_src)]
        if by_src[src]:
            picked.append(by_src[src].pop())
        i += 1
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=45)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=EVAL_DIR / "qa_pairs.draft.jsonl")
    args = ap.parse_args()

    with open(settings.chunks_path, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    llm = get_chat_model(settings.judge_provider, settings.judge_model)
    chain = PROMPT | llm

    rows, candidates = [], pick_chunks(chunks, args.n * 2, args.seed)
    for c in candidates:
        if len(rows) >= args.n:
            break
        try:
            raw = chain.invoke({"source": c["source"], "text": c["text"]}).content
            data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        except Exception as e:
            print(f"  skip ({type(e).__name__}) {c['source']}")
            time.sleep(2)
            continue
        if data.get("skip") or not data.get("question"):
            continue
        pages = locate_pages(settings.pdf_dir / c["source"], data.get("evidence", ""), c["page_start"], c["page_end"])
        rows.append({"id": f"q{len(rows) + 1:02d}", "question": data["question"], "answer": data["answer"],
                     "source": c["source"], "pages": pages, "evidence": data.get("evidence", ""),
                     "reviewed": False})
        print(f"  q{len(rows):02d} {c['source']} p.{pages}: {data['question']}")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(rows)} drafts -> {args.out}\nReview each against the PDF, then copy into qa_pairs.jsonl.")


if __name__ == "__main__":
    main()
