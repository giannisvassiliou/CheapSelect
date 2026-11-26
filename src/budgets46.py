#!/usr/bin/env python3
"""
budgets46.py: retrospective budgets on the grid {0, 10, ..., 100}% (b = 0 = re-rank none now included).
For each pair and selector: point budget (mean loss <= delta at b and every larger budget) and non-inferior budget
(95% bootstrap lower bound of selective - all > -delta at b and every larger budget, i.e. one-sided upper bound of the
loss < delta). Random: expected point budget, smallest b with (1-b)*mean gain <= delta at b and all larger b.
Usage: python budgets46.py amazon_testfull:full mag_testfull:full ... --out b46_main      -> b46_main.csv / .md
"""
import argparse
import csv

import numpy as np

from common46 import GRID, load, lower_ci_of_mean, order_budget_values, smallest_passing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+")
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--metric", default="MRR", choices=["MRR", "ndcg"])
    ap.add_argument("--out", default="b46")
    a = ap.parse_args()
    d = a.delta / 100
    rows, md = [], [f"# budgets46: grid includes b = 0; delta = {a.delta} {a.metric} point(s)\n",
                    "pair | gain | random(exp) | " + " | ".join(["WIG pt/NI", "tag pt/NI", "margin pt/NI", "NQC pt/NI", "top5 pt/NI", "termcov pt/NI", "oracle pt"])]
    for p in a.pairs:
        D = load(p, a.metric)
        g = D["g"]
        gbar = g.mean()
        rnd = GRID[smallest_passing([(1 - b / 100) * gbar <= d + 1e-12 for b in GRID])]
        res = {}
        sels = dict(D["sig"])
        sels["ORACLE"] = -g
        for s, sc in sels.items():
            L = order_budget_values(sc, g)
            pt = GRID[smallest_passing([v.mean() <= d + 1e-12 for v in L])]
            ni = GRID[smallest_passing([lower_ci_of_mean(-v) > -d for v in L[:-1]] + [True])]
            res[s] = (pt, ni)
            rows.append({"pair": p, "dataset": D["name"], "tag": D["tag"], "metric": a.metric, "delta": a.delta, "selector": s,
                         "point_budget": pt, "ni_budget": ni, "random_expected_budget": rnd,
                         "mean_gain": round(100 * gbar, 3), "n": D["n"]})
        f = lambda s: f"{res[s][0]}/{res[s][1]}" if s in res else "-"
        md.append(f"{p} | {100 * gbar:+.1f} | {rnd} | " + " | ".join(f(s) for s in ("WIG-like", "tag alarm", "BM25 margin", "NQC-like spread", "top5 vs rest", "term coverage top1"))
                  + f" | {res['ORACLE'][0]}")
    with open(a.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    open(a.out + ".md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
