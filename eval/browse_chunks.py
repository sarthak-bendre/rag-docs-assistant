"""Helper for writing QA pairs by hand: print random chunks (or keyword matches) with file/page.

  python -m eval.browse_chunks            # 5 random chunks
  python -m eval.browse_chunks torque     # chunks containing "torque"
"""
import json
import random
import sys

from src.config import settings

with open(settings.chunks_path, encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]

term = " ".join(sys.argv[1:]).lower()
pool = [c for c in chunks if term in c["text"].lower()] if term else random.sample(chunks, min(5, len(chunks)))
for c in pool[:10]:
    print(f"\n### {c['source']}  p.{c['page_start']}-{c['page_end']}\n{c['text'][:800]}")
print(f"\n({len(pool)} matching chunks)")
