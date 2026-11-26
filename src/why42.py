#!/usr/bin/env python3
"""
why42.py: WHY a selector beats random on some collection-re-ranker pairs and not on others.

Loss of selective re-ranking at budget b = (1-b) * mean gain of the SKIPPED queries.
  enrichment r(b) = mean gain of skipped queries / mean gain of all queries      (random: r = 1)
  required  r*(b) = delta / ((1-b) * mean gain)                                 (loss <= delta  <=>  r <= r*)
A selector skips a share b safely iff its enrichment is below the required one.
The script also decomposes the skipped queries' gain into the part from helped and from hurt queries,
and reports the skipped queries' first-stage quality (headroom), to show WHERE the gain of the skipped queries comes from.

Usage: python why42.py amazon_testfull:full mag_testfull:full prime_testfull:full dbpedia_test:dbp --out w42_stark_bge.md
Writes <out>.md and <out>.csv  (no GPU, seconds)
"""
import argparse
import csv
import json
import os

import numpy as np


def load(arg):
    folder, tag = arg.split(":")
    rows = [json.loads(l) for l in open(os.path.join(folder, f"selective_pq_{tag}.jsonl"))]
    bm = np.array([r["bm"]["MRR"] for r in rows], float)
    ce = np.array([r["ce"]["MRR"] for r in rows], float)
    sig = {"WIG": np.array([r["qpp"]["WIG-like"] for r in rows], float),
           "NQC": np.array([r["qpp"]["NQC-like spread"] for r in rows], float),
           "BM25 margin": np.array([r["margin"] for r in rows], float)}
    alarm = np.array([r["alarm"] for r in rows], float)
    orders = {k: np.argsort(v, kind="mergesort") for k, v in sig.items()}
    orders["tag alarm"] = np.lexsort((sig["BM25 margin"], -alarm))
    orders["ORACLE"] = np.argsort(-(ce - bm), kind="mergesort")
    return os.path.basename(folder.rstrip("/")), tag, bm, ce, orders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--out", default="why42.md")
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--budgets", type=int, nargs="+", default=[30, 50, 70])
    args = ap.parse_args()
    d = args.delta / 100
    out, rows = [], []

    def p(s=""):
        print(s)
        out.append(s)

    p("# why42: enrichment of the skipped queries (r = mean gain skipped / mean gain all; random r = 1; safe iff r <= r*)\n")
    for a in args.datasets:
        name, tag, bm, ce, orders = load(a)
        g = ce - bm
        n, gbar = len(g), g.mean()
        p(f"## {name} (tag {tag}; n={n}; mean gain {100 * gbar:+.2f}; helped {100 * (g > 0).mean():.0f}% hurt {100 * (g < 0).mean():.0f}%)")
        p(f"   {'selector':12s} {'b':>4s} {'r':>6s} {'r*':>6s} {'safe':>4s} | skipped: {'helped%':>7s} {'hurt%':>6s} "
          f"{'help mass':>9s} {'hurt mass':>9s} {'net':>7s} | {'BM25 MRR skipped':>16s} {'reranked':>8s}")
        for sel in ("random", "WIG", "tag alarm", "NQC", "BM25 margin", "ORACLE"):
            for b in args.budgets:
                k = int(round(n * b / 100))
                if sel == "random":
                    skip = np.ones(n, bool)  # expectation: skipped queries are a random sample = all queries
                    rr = np.ones(n, bool)
                else:
                    o = orders[sel]
                    skip = np.zeros(n, bool)
                    skip[o[k:]] = True
                    rr = ~skip
                gs = g[skip]
                r = gs.mean() / gbar if gbar > 0 else float("nan")
                rstar = d / ((1 - b / 100) * gbar) if gbar > 0 else float("nan")
                help_mass = gs[gs > 0].sum() / len(gs)
                hurt_mass = gs[gs < 0].sum() / len(gs)
                safe = (1 - b / 100) * gs.mean() <= d + 1e-12
                p(f"   {sel:12s} {b:3d}% {r:6.2f} {rstar:6.2f} {'yes' if safe else 'no':>4s} | "
                  f"{100 * (gs > 0).mean():7.0f} {100 * (gs < 0).mean():6.0f} {100 * help_mass:+9.2f} {100 * hurt_mass:+9.2f} {100 * gs.mean():+7.2f} | "
                  f"{100 * bm[skip].mean():16.1f} {100 * bm[rr].mean():8.1f}")
                rows.append({"dataset": name, "tag": tag, "selector": sel, "budget": b, "mean_gain": round(100 * gbar, 3),
                             "r": round(r, 4), "r_required": round(rstar, 4), "safe": bool(safe),
                             "skipped_helped_pct": round(100 * (gs > 0).mean(), 2), "skipped_hurt_pct": round(100 * (gs < 0).mean(), 2),
                             "skipped_help_mass": round(100 * help_mass, 3), "skipped_hurt_mass": round(100 * hurt_mass, 3),
                             "skipped_net_gain": round(100 * gs.mean(), 3), "bm25_mrr_skipped": round(100 * bm[skip].mean(), 2),
                             "bm25_mrr_reranked": round(100 * bm[rr].mean(), 2)})
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
