#!/usr/bin/env python3
"""
ndcg43.py: write nDCG@10 copies of v43 per-query files, so every statistics script runs unchanged on nDCG@10.
For each folder:tag it writes <folder>/selective_pq_<tag>ndcg.jsonl in which the "MRR" fields hold nDCG@10
(first stage and re-ranker); all selector signals are copied unchanged.
Usage: python ndcg43.py mag_testfull:bge43 beir_fiqa:bge43 ...      then e.g.  sel_stats40.py mag_testfull --tag bge43ndcg
"""
import json
import os
import sys

for a in sys.argv[1:]:
    folder, tag = a.split(":")
    src = os.path.join(folder, f"selective_pq_{tag}.jsonl")
    if not os.path.exists(src):
        print(f"missing {src}")
        continue
    rows = [json.loads(l) for l in open(src)]
    if "ndcg" not in rows[0]:
        print(f"{src}: no nDCG field (not a v43 run)")
        continue
    with open(os.path.join(folder, f"selective_pq_{tag}ndcg.jsonl"), "w") as f:
        for r in rows:
            r = dict(r)
            r["bm"] = dict(r["bm"], MRR=r["ndcg"]["bm"])
            r["ce"] = dict(r["ce"], MRR=r["ndcg"]["ce"])
            f.write(json.dumps(r) + "\n")
    print(f"wrote {folder}/selective_pq_{tag}ndcg.jsonl ({len(rows)} rows)")
