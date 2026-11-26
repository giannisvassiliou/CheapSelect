#!/usr/bin/env python3
"""
calib43.py: choosing the re-ranking THRESHOLD from n labelled queries, with risk-control baselines.

Deployment model (online): a query is re-ranked iff its proxy score is <= tau (low confidence first); tau is chosen
on n labelled calibration queries and then applied unchanged to new queries.
Loss of a query = its re-ranker gain if it is NOT re-ranked, else 0   (so loss <= delta  <=>  non-inferior;
positive loss = worse than re-ranking everything; this is the sign used throughout).
Candidate thresholds = the proxy-score quantiles at shares 10..100% of the calibration set.
Rules (pick the most permissive threshold, i.e. the fewest re-ranked queries, that passes, scanning from 100% down
and stopping at the first failure, so all larger budgets also pass):
  point   : mean calibration loss <= delta
  UCB95   : one-sided 95% upper confidence bound (normal approx.) of the loss <= delta
  LTT     : Learn-then-Test (Angelopoulos et al. 2021), fixed-sequence testing of H0: risk > delta with Hoeffding
            p-values for losses in [-1, 1]; guarantees P(risk > delta) <= alpha = 0.05 under exchangeability
  CRC     : conformal risk control (Angelopoulos et al. 2022) on the clipped loss max(gain,0)*skipped in [0,1]
            (monotone in tau): (n/(n+1)) * mean + 1/(n+1) <= delta; controls the EXPECTED clipped loss
Evaluation per split: held-out loss (other queries) AND population loss (whole collection, the distribution the
guarantee refers to); violation = loss > delta. Also the share re-ranked on held-out queries.
Selectors: WIG, tag alarm, random score.   n = 100, 200, 500 (500 only for collections with >= 1500 queries).

Usage: python calib43.py mag_testfull:full prime_testfull:full ... --out c43_stark_bge.md   (CPU, ~1 min)
"""
import argparse
import csv
import json
import os
from statistics import NormalDist

import numpy as np

SHARES = np.arange(10, 101, 10)


def load(arg):
    folder, tag = arg.split(":")
    rows = [json.loads(l) for l in open(os.path.join(folder, f"selective_pq_{tag}.jsonl"))]
    g = np.array([r["ce"]["MRR"] - r["bm"]["MRR"] for r in rows], float)
    wig = np.array([r["qpp"]["WIG-like"] for r in rows], float)
    margin = np.array([r["margin"] for r in rows], float)
    alarm = np.array([r["alarm"] for r in rows], float)
    # one score per selector, re-rank if score <= tau ; tag alarm: high alarm first, then small margin
    mr = (np.argsort(np.argsort(margin, kind="mergesort")) + 0.5) / len(margin)
    tag_score = -alarm + mr  # alarm is an integer; margin rank breaks ties
    return os.path.basename(folder.rstrip("/")), tag, g, {"WIG": wig, "tag alarm": tag_score}


def choose_tau(sc, g, delta, rule, alpha=0.05):
    """sc, g: calibration scores and gains. Returns tau (re-rank iff score <= tau); +inf = re-rank all."""
    n = len(g)
    z = NormalDist().inv_cdf(1 - alpha)
    tau = np.inf
    for share in SHARES[::-1][1:]:          # 90, 80, ..., 10 ; 100% always passes
        t = np.quantile(sc, share / 100.0)
        skip = sc > t
        loss = np.where(skip, g, 0.0)
        m = loss.mean()
        if rule == "point":
            ok = m <= delta
        elif rule == "UCB95":
            ok = m + z * loss.std(ddof=1) / np.sqrt(n) <= delta
        elif rule == "LTT":
            p = 1.0 if m >= delta else float(np.exp(-2 * n * (delta - m) ** 2 / 4.0))
            ok = p <= alpha
        elif rule == "CRC":
            lc = np.where(skip, np.maximum(g, 0.0), 0.0)
            ok = (n / (n + 1)) * lc.mean() + 1.0 / (n + 1) <= delta
        else:
            raise ValueError(rule)
        if ok:
            tau = t
        else:
            break
    return tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--out", default="calib43.md")
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--ncal", type=int, nargs="+", default=[100, 200, 500])
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()
    delta = args.delta / 100
    rules = ("point", "UCB95", "LTT", "CRC")
    out, rows = [], []

    def p(s=""):
        print(s)
        out.append(s)

    p(f"# calib43: threshold calibration on n labelled queries (delta = {args.delta}; {args.reps} splits; loss > 0 = worse than re-ranking all)\n")
    p("cell = share re-ranked on held-out % | P(held-out loss > delta) | P(population loss > delta)\n")
    for a in args.datasets:
        name, tag, g, sig = load(a)
        N = len(g)
        rng0 = np.random.default_rng(5)
        sig = dict(sig)
        sig["random"] = rng0.random(N)
        p(f"## {name} (tag {tag}; N={N}; mean gain {100 * g.mean():+.2f})")
        p(f"   {'selector':10s} {'n':>4s}  " + "  ".join(f"{r:>22s}" for r in rules))
        for sel, sc in sig.items():
            for n in args.ncal:
                if n >= N - 100 or (n >= 500 and N < 1500):
                    continue
                rng = np.random.default_rng(42)
                acc = {r: [] for r in rules}
                for _ in range(args.reps):
                    perm = rng.permutation(N)
                    C, H = perm[:n], perm[n:]
                    for r in rules:
                        tau = choose_tau(sc[C], g[C], delta, r)
                        skipH = sc[H] > tau
                        lh = float(np.where(skipH, g[H], 0).mean())
                        lp = float(np.where(sc > tau, g, 0).mean())
                        acc[r].append((100 * (1 - skipH.mean()), lh > delta + 1e-12, lp > delta + 1e-12))
                cells = []
                for r in rules:
                    x = np.array(acc[r], float)
                    share, vh, vp = x[:, 0].mean(), x[:, 1].mean(), x[:, 2].mean()
                    cells.append(f"{share:5.0f}% {100 * vh:6.1f}% {100 * vp:6.1f}%")
                    rows.append({"dataset": name, "tag": tag, "selector": sel, "n_cal": n, "rule": r,
                                 "share_reranked": round(share, 1), "saving": round(100 - share, 1),
                                 "p_violation_heldout": round(vh, 3), "p_violation_population": round(vp, 3)})
                p(f"   {sel:10s} {n:4d}  " + "  ".join(f"{c:>22s}" for c in cells))
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
