#!/usr/bin/env python3
"""
calib42.py: choosing the re-ranking budget from n labelled queries -- point rule vs conservative rules.
Offline: reads <data>/selective_pq_<tag>.jsonl. No GPU.

For each split: n labelled (calibration) queries C, the rest H. The selector orders queries without labels; the budget b
is chosen on C, then applied to H and the held-out loss  mean_H(selective) - mean_H(rerank all)  is measured.
Rules (b = smallest grid budget such that the condition holds at b and at every larger budget):
  point     : calibration mean loss >= -delta                           (the rule of sel_stats40 section J)
  LCB95     : one-sided 95% lower confidence bound of the loss >= -delta (normal approx., per budget)
  LCB-simul : the same with a Bonferroni-corrected bound over the 9 budgets (one-sided 1 - 0.05/9)
A rule is reliable if P(held-out loss < -delta) stays near or below its nominal 5%.
Selectors: WIG, tag alarm, and RANDOM order (so random's budget is also estimated from labels, with the gain unknown).

Usage:  python calib42.py amazon_testfull:full mag_testfull:full prime_testfull:full dbpedia_test:dbp --out c42_stark_bge.md
Writes <out>.md and <out>.csv
"""
import argparse
import csv
import json
import os
from statistics import NormalDist

import numpy as np

GRID = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])


def load(arg):
    folder, tag = arg.split(":")
    rows = [json.loads(l) for l in open(os.path.join(folder, f"selective_pq_{tag}.jsonl"))]
    bm = np.array([r["bm"]["MRR"] for r in rows], float)
    ce = np.array([r["ce"]["MRR"] for r in rows], float)
    margin = np.array([r["margin"] for r in rows], float)
    alarm = np.array([r["alarm"] for r in rows], float)
    wig = np.array([r["qpp"]["WIG-like"] for r in rows], float)
    return os.path.basename(folder.rstrip("/")), tag, bm, ce, margin, alarm, wig


def loss_vectors(order_idx, bm, ce):
    """order_idx: query indices (a subset) in selection order. Returns, per grid budget, the per-query
    differences (selective - all) over that subset: shape (len(GRID), len(subset))."""
    n = len(order_idx)
    g = ce[order_idx] - bm[order_idx]
    out = np.empty((len(GRID), n))
    for i, b in enumerate(GRID):
        k = int(round(n * b / 100))
        d = -g.copy()          # not re-ranked: lose the gain
        d[:k] = 0.0            # re-ranked: identical to rerank-all
        out[i] = d
    return out


def choose(D, delta, z):
    """D: (budgets, n) per-query differences on the calibration set. z: one-sided quantile (0 = point rule)."""
    n = D.shape[1]
    lcb = D.mean(axis=1) - z * D.std(axis=1, ddof=1) / np.sqrt(n)
    ok = lcb >= -delta
    chosen = 100
    for i in range(len(GRID) - 1, -1, -1):
        if ok[i]:
            chosen = GRID[i]
        else:
            break
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--out", default="calib42.md")
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--ncal", type=int, nargs="+", default=[100, 200, 500])
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()
    delta = args.delta / 100
    rules = {"point": 0.0, "LCB95": NormalDist().inv_cdf(0.95), "LCB-simul": NormalDist().inv_cdf(1 - 0.05 / len(GRID))}
    out, rows = [], []

    def p(s=""):
        print(s)
        out.append(s)

    p(f"# calib42: budget calibration on n labelled queries (delta = {args.delta} MRR point; {args.reps} splits)\n")
    p("columns: mean chosen budget % | mean held-out loss (MRR points) | P(held-out loss < -delta) | mean saving %\n")
    for a in args.datasets:
        name, tag, bm, ce, margin, alarm, wig = load(a)
        n = len(bm)
        p(f"## {name} (tag {tag}; n={n}; mean gain {100 * (ce - bm).mean():+.1f})")
        p(f"   {'selector':10s} {'n_cal':>5s}  " + "  ".join(f"{r:>30s}" for r in rules))
        for sel in ("WIG", "tag alarm", "random"):
            for nc in args.ncal:
                if nc > n - 100:
                    continue
                rng = np.random.default_rng(42)
                res = {r: [] for r in rules}
                for _ in range(args.reps):
                    perm = rng.permutation(n)
                    C, H = perm[:nc], perm[nc:]

                    def order(idx):
                        if sel == "WIG":
                            return idx[np.argsort(wig[idx], kind="mergesort")]
                        if sel == "tag alarm":
                            return idx[np.lexsort((margin[idx], -alarm[idx]))]
                        return rng.permutation(idx)
                    DC = loss_vectors(order(C), bm, ce)
                    DH = loss_vectors(order(H), bm, ce)
                    for r, z in rules.items():
                        b = choose(DC, delta, z)
                        lh = 0.0 if b == 100 else float(DH[np.where(GRID == b)[0][0]].mean())
                        res[r].append((b, lh))
                cells = []
                for r in rules:
                    bs = np.array([x[0] for x in res[r]])
                    ls = np.array([x[1] for x in res[r]])
                    viol = float(np.mean(ls < -delta))
                    cells.append(f"{bs.mean():5.0f}% {100 * ls.mean():+6.2f} {100 * viol:5.0f}% {100 - bs.mean():5.0f}%")
                    rows.append({"dataset": name, "tag": tag, "selector": sel, "n_cal": nc, "rule": r,
                                 "mean_budget": round(float(bs.mean()), 1), "mean_heldout_loss": round(float(100 * ls.mean()), 3),
                                 "p_violation": round(viol, 3), "mean_saving": round(float(100 - bs.mean()), 1)})
                p(f"   {sel:10s} {nc:5d}  " + "  ".join(f"{c:>30s}" for c in cells))
        p()
    with open(args.out, "w") as f:
        f.write("\n".join(out) + "\n")
    with open(os.path.splitext(args.out)[0] + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
