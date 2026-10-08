"""Download the 20 public-domain manuals listed in data/manuals.tsv into data/pdfs/.

  python scripts/download_manuals.py
"""
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "pdfs"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = (ROOT / "data" / "manuals.tsv").read_text(encoding="utf-8").splitlines()
    for line in filter(None, lines):
        name, url = line.split("\t")
        dest = OUT / name
        if dest.exists():
            print(f"  skip  {name} (already downloaded)")
            continue
        try:
            r = requests.get(url, headers=HEADERS, timeout=180)
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                raise ValueError("response is not a PDF")
            dest.write_bytes(r.content)
            print(f"  ok    {name} ({len(r.content) / 1e6:.1f} MB)")
        except Exception as e:
            print(f"  FAIL  {name}: {e}")


if __name__ == "__main__":
    main()
