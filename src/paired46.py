#!/usr/bin/env python3
"""
paired46.py: compare the re-rankers on EXACTLY the queries of the Qwen3-Reranker-8B samples (review item 8).
Joins per-query files by query id (qid_str). For each collection: n, mean gain of every re-ranker on the shared ids,
paired difference Qwen - X with a 95% bootstrap interval, and WIG's point budget (grid incl. 0) per re-ranker on those ids.
Usage: python paired46.py --qwen qwen8b --others mini43 bge43 monot5 \
          mag_testfull prime_testfull amazon_testfull dbpedia_test beir_scifact beir_nfcorpus beir_scidocs --out p46
"""
import argparse
import csv
import os

import numpy as np

from common46 import GRID, load, order_budget_values, smallest_passing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--qwen", default="qwen8b")
    ap.add_argument("--others", nargs="+", default=["mini43", "bge43", "monot5"])
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--out", default="p46")
    a = ap.parse_args()
    d = a.delta / 100
    rows, md = [], ["# paired46: re-rankers on the Qwen sample ids (MRR points; 95% bootstrap CI of the paired difference)\n"]
    rng = np.random.default_rng(7)
    for f in a.folders:
        tags = [a.qwen] + [t for t in a.others if os.path.exists(os.path.join(f, f"selective_pq_{t}.jsonl"))]
        Ds = {t: load(f"{f}:{t}") for t in tags}
        ids = set(Ds[a.qwen]["qid"])
        for t in tags[1:]:
            ids &= set(Ds[t]["qid"])
        ids = sorted(ids)
        if not ids:
            md.append(f"## {f}: no shared ids\n")
            continue
        G, W = {}, {}
        for t, D in Ds.items():
            pos = {q: i for i, q in enumerate(D["qid"])}
            ii = np.array([pos[q] for q in ids])
            G[t] = D["g"][ii]
            W[t] = D["sig"]["WIG-like"][ii]
        md.append(f"## {f} (shared ids: {len(ids)})")
        for t in tags:
            L = order_budget_values(W[t], G[t])
            pt = GRID[smallest_passing([v.mean() <= d + 1e-12 for v in L])]
            line = f"   {t:10s} mean gain {100 * G[t].mean():+6.2f}   WIG point budget {pt:3d}%"
            row = {"collection": f, "reranker": t, "n_shared": len(ids), "mean_gain": round(100 * G[t].mean(), 3), "wig_point_budget": int(pt)}
            if t != a.qwen:
                diff = G[a.qwen] - G[t]
                bs = diff[rng.integers(0, len(diff), size=(2000, len(diff)))].mean(axis=1)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                line += f"   Qwen - {t}: {100 * diff.mean():+6.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]"
                row.update({"qwen_minus": round(100 * diff.mean(), 3), "ci_lo": round(100 * lo, 3), "ci_hi": round(100 * hi, 3)})
            md.append(line)
            rows.append(row)
        md.append("")
    open(a.out + ".md", "w").write("\n".join(md) + "\n")
    keys = sorted({k for r in rows for k in r}, key=lambda k: ["collection", "reranker", "n_shared", "mean_gain", "wig_point_budget", "qwen_minus", "ci_lo", "ci_hi"].index(k))
    with open(a.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("\n".join(md))


if __name__ == "__main__":
    main()
