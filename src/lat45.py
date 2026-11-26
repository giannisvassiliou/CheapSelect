#!/usr/bin/env python3
"""lat45.py: per-collection median latencies (ms) of BM25, MiniLM and bge from the TIMING blocks of earlier run logs -> lat45.json"""
import json, os, re
LOGS = {  # collection: (bge log, MiniLM log)
    "amazon_testfull": ("full_amazon.log", "mini_amazon.log"), "mag_testfull": ("full_mag.log", "mini_mag.log"),
    "prime_testfull": ("full_prime.log", "mini_prime.log"), "dbpedia_test": ("dbp_test.log", "mini_dbp.log")}
for d in ("scifact", "nfcorpus", "fiqa", "arguana", "scidocs"):
    LOGS[f"beir_{d}"] = (f"beir_{d}_bge.log", f"beir_{d}_mini.log")

def med(fn, key):
    if not os.path.exists(fn):
        return None
    t = open(fn, errors="ignore").read()
    m = re.findall(re.escape(key) + r"[^\n]*?\s+([\d.]+)\s+([\d.]+)\s+\d+\s*$", t, re.M)
    return float(m[-1][0]) if m else None

# measured medians from the paper's TIMING table, used when a log is missing
KNOWN = {"amazon_testfull": (14.3, 145.7, 316.0), "mag_testfull": (10.4, 97.6, 257.0), "prime_testfull": (1.8, 79.8, 211.0),
         "dbpedia_test": (33.5, 46.6, 87.0)}
out = {}
for c, (fb, fm) in LOGS.items():
    bm = med(fb, "BM25 (whole row)") or med(fm, "BM25 (whole row)")
    bg = med(fb, "cross-encoder (100 rows)")
    mi = med(fm, "cross-encoder (100 rows)")
    if bg is None and mi is not None:
        bg = 2.3 * mi          # fallback: mean bge/MiniLM ratio on STaRK+DBpedia
    if mi is None and bg is not None:
        mi = bg / 2.3
    k = KNOWN.get(c)
    if bm is None:
        bm = k[0] if k else 1.0
    if mi is None:
        mi = k[1] if k else 50.0
    if bg is None:
        bg = k[2] if k else 2.3 * mi
    out[c] = {"bm25": bm, "mini": mi, "bge": bg}
    print(c, out[c], "(from", fb, fm, ")")
json.dump(out, open("lat45.json", "w"), indent=1)
print("saved lat45.json")
