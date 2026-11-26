#!/usr/bin/env python3
"""
qpp_gain_corr.py: does each label-free signal track (a) how good BM25 is, or (b) how much the re-ranker gains?

For every dataset and signal it prints Spearman correlations with
  quality = BM25 MRR of the query
  gain    = cross-encoder MRR - BM25 MRR of the query
Signals are oriented so that HIGHER = "BM25 is fine, skip the re-ranker" (alarm is negated).
Expected if the paper's explanation holds: strong positive corr with quality everywhere,
but clearly negative corr with gain only where selection works (MAG, Prime).

Usage:
  python qpp_gain_corr.py amazon_testfull:full mag_testfull:full prime_testfull:full dbpedia_test:dbp
"""
import json
import os
import sys

import numpy as np


def ranks(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    # average ranks for ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    return (sums / counts)[inv]


def spearman(a, b, n_boot=2000, seed=0):
    ra, rb = ranks(a), ranks(b)
    rho = float(np.corrcoef(ra, rb)[0, 1])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ra), size=(n_boot, len(ra)))
    bs = [np.corrcoef(ra[i], rb[i])[0, 1] for i in idx]
    return rho, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    out = []
    for arg in sys.argv[1:]:
        folder, tag = arg.split(":") if ":" in arg else (arg, "full")
        rows = [json.loads(l) for l in open(os.path.join(folder, f"selective_pq_{tag}.jsonl"))]
        bm = np.array([r["bm"]["MRR"] for r in rows])
        ce = np.array([r["ce"]["MRR"] for r in rows])
        gain = ce - bm
        sig = {"BM25 margin": np.array([r["margin"] for r in rows]),
               "tag alarm (negated)": -np.array([r["alarm"] for r in rows])}
        for name in (rows[0].get("qpp") or {}):
            sig[name] = np.array([r["qpp"][name] for r in rows])
        name_ds = os.path.basename(folder.rstrip("/"))
        out.append(f"## {name_ds}  (n={len(rows)}; mean gain {100 * gain.mean():+.1f} MRR; "
                   f"queries helped {100 * (gain > 0).mean():.0f}%, hurt {100 * (gain < 0).mean():.0f}%)")
        out.append(f"   corr(BM25 quality, gain) = {spearman(bm, gain)[0]:+.2f}")
        out.append(f"   {'signal':22s} {'rho(quality) [95% CI]':>26s} {'rho(gain) [95% CI]':>26s}")
        for sn, sv in sig.items():
            q = spearman(sv, bm)
            g = spearman(sv, gain)
            out.append(f"   {sn:22s} {q[0]:+.2f} [{q[1]:+.2f},{q[2]:+.2f}]      {g[0]:+.2f} [{g[1]:+.2f},{g[2]:+.2f}]")
        out.append("")
    txt = "\n".join(out)
    print(txt)
    with open("qpp_gain_corr.md", "w") as f:
        f.write(txt + "\n")
    print("saved qpp_gain_corr.md")


if __name__ == "__main__":
    main()
