#!/usr/bin/env python3
"""
regime43.py: can the regime (re-ranker useless / selection pays / too strong to skip) be predicted WITHOUT labels?

For every collection-re-ranker pair it computes label-free statistics of the re-ranker's behaviour on a small
unlabelled sample (default 100 queries, 200 resamples), which only requires running the re-ranker on that sample:
  dis1  : share of queries where the re-ranker's top-1 differs from BM25's top-1   (v43 files; else top-3 probe)
  rho10 : mean rank correlation between re-ranker and BM25 order on BM25's top-10  (v43 files only)
and compares them, across pairs, with the label-based quantities the guidelines need: mean gain g, WIG's point budget,
and WIG's saving beyond random. Spearman correlations over pairs; the regime is
  're-ranker adds little' (random budget <= 20%), 'selection pays' (WIG saves >= 20 points beyond random), 'selection adds little' (otherwise).

Usage: python regime43.py amazon_testfull:full mag_testfull:full ... beir_scifact:monot5 --out g43.md
"""
import argparse
import json
import os

import numpy as np

B = np.arange(10, 101, 10)


def spearman(a, b):
    ra = np.argsort(np.argsort(a, kind="mergesort"))
    rb = np.argsort(np.argsort(b, kind="mergesort"))
    return float(np.corrcoef(ra, rb)[0, 1])


def point_budget(order, g, d):
    n = len(g)
    means = []
    for b in B:
        k = int(round(n * b / 100))
        skip = np.ones(n, bool)
        skip[order[:k]] = False
        means.append(-np.where(skip, g, 0).mean())
    return next(B[j] for j in range(len(B)) if all(m >= -d - 1e-12 for m in means[j:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--out", default="regime43.md")
    args = ap.parse_args()
    d = args.delta / 100
    R = []
    out = []

    def p(s=""):
        print(s)
        out.append(s)

    for a in args.datasets:
        folder, tag = a.split(":")
        rows = [json.loads(l) for l in open(os.path.join(folder, f"selective_pq_{tag}.jsonl"))]
        g = np.array([r["ce"]["MRR"] - r["bm"]["MRR"] for r in rows])
        wig = np.array([r["qpp"]["WIG-like"] for r in rows])
        if "cef" in rows[0]:
            dis = 1 - np.array([r["cef"]["agree_top1"] for r in rows], float)
            rho = np.array([r["cef"]["rho10"] for r in rows], float)
        else:
            dis = 1 - np.array([r["probe3_agree"] for r in rows], float)
            rho = None
        n = len(g)
        gbar = g.mean()
        rp = next(b for b in B if all((1 - int(round(n * bb / 100)) / n) * gbar <= d + 1e-12 for bb in B if bb >= b))
        wp = point_budget(np.argsort(wig, kind="mergesort"), g, d)
        rng = np.random.default_rng(0)
        smp = [rng.choice(n, size=min(args.sample, n), replace=False) for _ in range(args.reps)]
        dis_s = np.array([dis[s].mean() for s in smp])
        rho_s = np.array([rho[s].mean() for s in smp]) if rho is not None else None
        regime = "re-ranker adds little" if rp <= 20 else ("selection pays" if rp - wp >= 20 else "selection adds little")
        R.append({"pair": f"{os.path.basename(folder)}:{tag}", "gain": 100 * gbar, "dis": dis.mean(), "dis_lo": np.percentile(dis_s, 5),
                  "dis_hi": np.percentile(dis_s, 95), "rho": (rho.mean() if rho is not None else np.nan), "rand_b": rp, "wig_b": wp,
                  "save": rp - wp, "regime": regime})
    p(f"# regime43: label-free re-ranker statistics on {args.sample}-query samples vs label-based quantities (delta = {args.delta})\n")
    p(f"   {'pair':28s} {'gain':>6s} {'dis1':>6s} {'90% range (sample)':>19s} {'rho10':>6s} {'rand b':>6s} {'WIG b':>6s} {'save':>5s}  regime")
    for r in sorted(R, key=lambda x: x["dis"]):
        p(f"   {r['pair']:28s} {r['gain']:6.1f} {r['dis']:6.2f} {r['dis_lo']:9.2f}-{r['dis_hi']:4.2f}     {r['rho']:6.2f} {r['rand_b']:6d} {r['wig_b']:6d} {r['save']:5d}  {r['regime']}")
    p()
    dis = np.array([r["dis"] for r in R]); gain = np.array([r["gain"] for r in R]); save = np.array([r["save"] for r in R])
    p(f"Spearman over {len(R)} pairs:  dis1 vs mean gain {spearman(dis, gain):+.2f};  dis1 vs WIG budget {spearman(dis, [r['wig_b'] for r in R]):+.2f};"
      f"  dis1 vs saving beyond random {spearman(dis, save):+.2f}")
    if not any(np.isnan(r["rho"]) for r in R):
        rho = np.array([r["rho"] for r in R])
        p(f"                  rho10 vs mean gain {spearman(rho, gain):+.2f};  rho10 vs saving beyond random {spearman(rho, save):+.2f}")
    with open(args.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
