"""Step 1 - Ingest: PDF -> cleaned text -> overlapping token chunks with file/page metadata."""
import bisect
import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf as fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings


@dataclass
class Chunk:
    id: str
    text: str
    source: str  # PDF file name
    page_start: int  # 1-based
    page_end: int

    def to_dict(self) -> dict:
        return asdict(self)


def clean(text: str) -> str:
    text = text.replace("­", "")  # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    # Unwrap hard line breaks from narrow PDF columns, but keep paragraph breaks,
    # breaks after a colon, and breaks before bullets / numbered items.
    text = re.sub(r"(?<=[^\n:])\n(?!\n|[•●▪•\-–*]|\(?\d{1,2}[.)]\s|\(?[a-z][.)]\s)", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_pdf(path: Path) -> tuple[str, list[int]]:
    """Return the full document text and the character offset where each page starts."""
    parts, page_offsets, pos = [], [], 0
    with fitz.open(path) as doc:
        for page in doc:
            page_offsets.append(pos)
            txt = clean(page.get_text("text")) + "\n\n"
            parts.append(txt)
            pos += len(txt)
    return "".join(parts), page_offsets


def make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=settings.chunk_tokens,
        chunk_overlap=settings.chunk_overlap,
    )


def chunk_pdf(path: Path, splitter=None) -> list[Chunk]:
    splitter = splitter or make_splitter()
    text, page_offsets = read_pdf(path)
    if not text.strip():
        print(f"  ! no extractable text in {path.name} (scanned PDF? needs OCR) - skipped")
        return []
    chunks, start = [], -1
    for seq, piece in enumerate(splitter.split_text(text)):
        # Locate chunks strictly left-to-right so repeated text (headers, boilerplate)
        # can't map a chunk back to an earlier page.
        found = text.find(piece, start + 1)
        start = found if found != -1 else start + 1
        end = start + len(piece) - 1
        p_start = bisect.bisect_right(page_offsets, start)
        p_end = bisect.bisect_right(page_offsets, end)
        cid = hashlib.md5(f"{path.name}:{seq}".encode()).hexdigest()[:12]
        chunks.append(Chunk(cid, piece, path.name, p_start, p_end))
    return chunks


def chunk_folder(pdf_dir: Path = settings.pdf_dir) -> list[Chunk]:
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {pdf_dir}")
    splitter = make_splitter()
    out = []
    for p in pdfs:
        cs = chunk_pdf(p, splitter)
        print(f"  {p.name}: {len(cs)} chunks")
        out.extend(cs)
    return out
