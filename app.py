"""Step 5 - Interface: Streamlit chat page.   Run:  streamlit run app.py  (or start.bat)"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")  # fewer threads = less reserved memory (start.bat also sets these)

import html
import re
import time
from collections import Counter

import pymupdf as fitz
import requests
import streamlit as st

from src.config import settings
from src.generate import answer
from src.llm import get_chat_model
from src.retrieve import Hit, Retriever

st.set_page_config(page_title="Manuals Assistant", page_icon="🔧", layout="centered")

DEFAULT_MODELS = {"ollama": "llama3.2:3b", "groq": "openai/gpt-oss-120b"}
EXAMPLES = [
    ("Lockout/tagout", "What are the steps of a lockout/tagout procedure?"),
    ("Transformer oil", "How often should transformer oil be sampled?"),
    ("Pump cavitation", "What causes cavitation in centrifugal pumps?"),
    ("Electrical PPE", "What PPE is required for working on energized electrical equipment?"),
]
STOPWORDS = set("what which when where how does is are the for with from that this into should be of a an to in on and or required".split())

CSS = """
<style>
.block-container {max-width: 940px; padding-top: 2.2rem;}
[data-testid="stSidebar"] {border-right: 1px solid #1F2937;}
[data-testid="stSidebar"] h3 {font-size: .78rem; letter-spacing: .08em; text-transform: uppercase;
  color: #94A3B8; margin: 1.1rem 0 .3rem;}

.hero {border: 1px solid #1F2937; border-radius: 16px; padding: 22px 26px; margin-bottom: 18px;
  background: radial-gradient(1200px 200px at 0% 0%, rgba(245,158,11,.14), transparent 60%), #0F172A;}
.hero-top {display: flex; align-items: center; gap: 14px;}
.hero-icon {width: 46px; height: 46px; border-radius: 12px; display: grid; place-items: center;
  background: linear-gradient(135deg, #F59E0B, #B45309); font-size: 24px;}
.hero h1 {font-size: 1.55rem; font-weight: 700; margin: 0; padding: 0; line-height: 1.2;}
.hero p {color: #94A3B8; margin: 4px 0 0; font-size: .92rem;}
.stats {display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px;}
.stat {border: 1px solid #1F2937; background: #0B1120; border-radius: 999px; padding: 4px 12px;
  font-size: .78rem; color: #CBD5E1;}
.stat b {color: #F59E0B; font-weight: 600;}

.empty-title {color: #94A3B8; font-size: .85rem; margin: 6px 0 8px;}

.cite {display: inline-block; min-width: 1.35em; text-align: center; font-size: .72em; font-weight: 700;
  padding: 0 .35em; margin: 0 .12em; border-radius: 6px; vertical-align: 2px;
  color: #F59E0B; background: rgba(245,158,11,.12); border: 1px solid rgba(245,158,11,.35);}

.notfound {border: 1px solid rgba(248,113,113,.35); background: rgba(248,113,113,.08);
  border-radius: 12px; padding: 12px 16px; color: #FCA5A5;}
.meta {color: #64748B; font-size: .76rem; margin-top: 6px;}

.src {border: 1px solid #1F2937; border-radius: 12px; padding: 12px 14px; margin-bottom: 10px; background: #0F172A;}
.src-head {display: flex; align-items: center; gap: 10px;}
.src-num {flex: none; width: 26px; height: 26px; border-radius: 8px; display: grid; place-items: center;
  font-size: .78rem; font-weight: 700; color: #0B1120; background: #F59E0B;}
.src-title {font-weight: 600; font-size: .9rem; color: #E5E7EB; flex: 1; min-width: 0;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;}
.src-org {font-size: .68rem; font-weight: 700; letter-spacing: .05em; color: #94A3B8;
  border: 1px solid #334155; border-radius: 6px; padding: 1px 6px;}
.page {flex: none; font-size: .74rem; color: #FCD34D; background: rgba(245,158,11,.1);
  border-radius: 999px; padding: 2px 10px;}
.bar {height: 4px; background: #1E293B; border-radius: 2px; margin: 10px 0 8px;}
.bar > div {height: 100%; border-radius: 2px; background: linear-gradient(90deg, #B45309, #F59E0B);}
.snippet {font-size: .85rem; line-height: 1.6; color: #CBD5E1;}
.snippet mark {background: rgba(245,158,11,.25); color: #FDE68A; padding: 0 2px; border-radius: 3px;}

.dot {display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px;}
.status {font-size: .8rem; color: #CBD5E1; padding: 6px 0;}
.doc-row {display: flex; justify-content: space-between; font-size: .78rem; color: #CBD5E1; padding: 3px 0;
  border-bottom: 1px dashed #1F2937;}
.doc-row span:last-child {color: #64748B;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------- cached resources ----------
@st.cache_resource(show_spinner="Loading search index and models…")
def load_retriever() -> Retriever:
    r = Retriever()
    _ = (r.embedder, r.reranker)  # load both models now so failures surface at startup, not mid-question
    return r


@st.cache_resource
def load_llm(provider: str, model: str):
    return get_chat_model(provider, model)


@st.cache_data
def page_png(source: str, page: int) -> bytes:
    with fitz.open(settings.pdf_dir / source) as doc:
        return doc[page - 1].get_pixmap(dpi=110).tobytes("png")


@st.cache_data(ttl=10, show_spinner=False)
def llm_online(provider: str) -> tuple[bool, str]:
    if provider == "groq":
        import os
        return (bool(os.getenv("GROQ_API_KEY")), "API key set" if os.getenv("GROQ_API_KEY") else "GROQ_API_KEY missing in .env")
    try:
        requests.get(f"{settings.ollama_base_url}/api/version", timeout=1.5).raise_for_status()
        return True, "Ollama running"
    except Exception:
        return False, "Ollama not running — start it or run start.bat"


# ---------- helpers ----------
def pretty_name(source: str) -> tuple[str, str]:
    """'OSHA_3120_Lockout_Tagout.pdf' -> ('OSHA', '3120 · Lockout Tagout')"""
    stem = re.sub(r"\.pdf$", "", source, flags=re.I)
    parts = stem.split("_")
    has_org = len(parts) > 1 and (parts[0].isupper() or parts[0] == "Army")
    org = parts[0].upper() if has_org else ""
    rest = parts[1:] if has_org else parts
    if len(rest) > 2 and rest[0] == "TM":
        title = f"TM {rest[1]} · " + " ".join(rest[2:])
    elif len(rest) > 1 and rest[0][0].isdigit():
        title = f"{rest[0]} · " + " ".join(rest[1:])
    else:
        title = " ".join(rest)
    return org, title.strip(" ·")


def query_terms(q: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9/\-]+", q.lower()) if t not in STOPWORDS and len(t) > 2]


def snippet_html(text: str, terms: list[str], width: int = 420) -> str:
    low = text.lower()
    pos = min((low.find(t) for t in terms if t in low), default=0)
    start = max(0, pos - width // 3)
    snip = text[start:start + width].replace("\n", " ")
    snip = ("…" if start else "") + snip + ("…" if start + width < len(text) else "")
    out = html.escape(snip)
    for t in sorted(set(terms), key=len, reverse=True):
        out = re.sub(rf"(?i)\b({re.escape(html.escape(t))}\w*)", r"<mark>\1</mark>", out)
    return out


def format_answer(text: str) -> str:
    safe = html.escape(text, quote=False)
    return re.sub(r"\[(\d{1,2})\]", r'<span class="cite">\1</span>', safe)


def render_sources(hits: list[Hit], question: str, show_pages: bool):
    terms = query_terms(question)
    scores = [h.score for h in hits]
    lo, hi = min(scores), max(scores)
    with st.expander(f"📚  {len(hits)} source passages", expanded=False):
        for i, h in enumerate(hits, 1):
            org, title = pretty_name(h.source)
            rel = 100 if hi == lo else int(15 + 85 * (h.score - lo) / (hi - lo))
            org_html = f'<span class="src-org">{html.escape(org)}</span>' if org else ""
            st.markdown(
                f'<div class="src"><div class="src-head"><div class="src-num">{i}</div>{org_html}'
                f'<div class="src-title" title="{html.escape(h.source)}">{html.escape(title)}</div>'
                f'<div class="page">{h.page_label}</div></div>'
                f'<div class="bar"><div style="width:{rel}%"></div></div>'
                f'<div class="snippet">{snippet_html(h.text, terms)}</div></div>',
                unsafe_allow_html=True,
            )
            if show_pages:
                st.image(page_png(h.source, h.page_start), caption=f"{h.source} — page {h.page_start}")


def render_assistant(msg: dict, show_pages: bool):
    if msg.get("not_found"):
        st.markdown('<div class="notfound">⚠️ <b>Not found in the documents.</b> '
                    'None of the indexed manuals cover this question.</div>', unsafe_allow_html=True)
    else:
        st.markdown(format_answer(msg["content"]), unsafe_allow_html=True)
    if msg.get("meta"):
        st.markdown(f'<div class="meta">{msg["meta"]}</div>', unsafe_allow_html=True)
    if msg.get("hits"):
        render_sources(msg["hits"], msg["question"], show_pages)


# ---------- sidebar ----------
with st.sidebar:
    st.markdown("### Model")
    provider = st.selectbox("Provider", ["ollama", "groq"],
                            index=0 if settings.llm_provider == "ollama" else 1,
                            format_func=lambda p: {"ollama": "Ollama (local)", "groq": "Groq (cloud)"}[p])
    model = st.text_input("Model name", DEFAULT_MODELS[provider], key=f"model_{provider}")
    ok, status = llm_online(provider)
    st.markdown(f'<div class="status"><span class="dot" style="background:{"#22C55E" if ok else "#EF4444"}"></span>'
                f'{status}</div>', unsafe_allow_html=True)

    st.markdown("### Retrieval")
    mode = st.segmented_control("Search mode", ["hybrid", "vector", "bm25"], default="hybrid",
                                help="Hybrid = semantic vectors + BM25 keywords, fused with RRF.") or "hybrid"
    use_rerank = st.toggle("Cross-encoder reranker", value=True)
    k = st.slider("Passages sent to the LLM", 1, 10, 5)
    show_pages = st.toggle("Show PDF page images", value=False)

    if settings.chunks_path.exists():
        st.markdown("### Knowledge base")
        retriever = load_retriever()
        counts = Counter(c["source"] for c in retriever.chunks)
        with st.expander(f"{len(counts)} documents"):
            rows = "".join(f'<div class="doc-row"><span>{html.escape(pretty_name(s)[1] or s)}</span><span>{n}</span></div>'
                           for s, n in sorted(counts.items()))
            st.markdown(rows, unsafe_allow_html=True)

    st.write("")
    if st.button("New conversation", icon=":material/refresh:", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------- main ----------
if not settings.chunks_path.exists():
    st.error("No index found. Put PDFs in `data/pdfs/` and run `python -m src.index`.")
    st.stop()

retriever = load_retriever()
n_docs = len({c["source"] for c in retriever.chunks})
st.markdown(
    f'<div class="hero"><div class="hero-top"><div class="hero-icon">🔧</div><div>'
    f'<h1>Maintenance Manuals Assistant</h1>'
    f'<p>Ask in plain English — answers come only from the indexed manuals, with page citations.</p></div></div>'
    f'<div class="stats"><span class="stat"><b>{n_docs}</b> documents</span>'
    f'<span class="stat"><b>{len(retriever.chunks):,}</b> passages</span>'
    f'<span class="stat">search: <b>{mode}{" + rerank" if use_rerank else ""}</b></span>'
    f'<span class="stat">model: <b>{html.escape(model)}</b></span></div></div>',
    unsafe_allow_html=True,
)

st.session_state.setdefault("messages", [])
AVATARS = {"user": ":material/person:", "assistant": ":material/engineering:"}

if not st.session_state.messages:
    st.markdown('<div class="empty-title">Try one of these</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (label, q) in enumerate(EXAMPLES):
        if cols[i % 2].button(f"**{label}**  \n{q}", key=f"ex{i}", use_container_width=True):
            st.session_state.pending = q
            st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar=AVATARS[m["role"]]):
        if m["role"] == "user":
            st.markdown(m["content"])
        else:
            render_assistant(m, show_pages)

question = st.chat_input("Ask about a procedure, limit, interval or part number…")
question = question or st.session_state.pop("pending", None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(question)
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        t0 = time.perf_counter()
        with st.spinner(f"Searching {n_docs} documents…"):
            try:
                ans = answer(question, retriever, load_llm(provider, model), k=k, mode=mode, rerank=use_rerank)
            except Exception as e:
                err = str(e)
                if "10061" in err or "refused" in err.lower() or "connect" in err.lower():
                    hint = ("**Can't reach Ollama.** Start it from the Start menu or run `start.bat`."
                            if provider == "ollama" else "**Can't reach Groq.** Check your internet connection.")
                elif provider == "groq" and ("401" in err or "404" in err or "api_key" in err.lower()):
                    hint = "**Groq rejected the request.** Check `GROQ_API_KEY` and the model name."
                else:
                    hint = "**Something failed while answering.** If memory is low, close other programs and restart the app."
                st.error(f"{hint}\n\nDetails: `{err[:300]}`")
                st.session_state.messages.pop()
                st.stop()
        msg = {"role": "assistant", "content": ans.text, "question": question,
               "not_found": ans.not_found, "hits": [] if ans.not_found else ans.hits,
               "meta": f"{time.perf_counter() - t0:.1f}s · {model} · {mode}{' + rerank' if use_rerank else ''}"}
        render_assistant(msg, show_pages)
    st.session_state.messages.append(msg)
