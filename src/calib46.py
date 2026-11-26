#!/usr/bin/env python3
"""
calib46.py: budget calibration with design and certification separated (review items 2, 4, 5, 6, 10).

Per split (default 200 splits, seed 42):
  C  = n labelled certification queries (the only labels used, except in --mode labels where n is split, see below)
  Dp = unlabelled design pool: 30% of the remaining queries (at least 100); only proxy SCORES are used, never labels
  T  = the other remaining queries: untouched test queries (realized share, loss and violation are measured here)
Candidate policies (fixed before looking at C): re-rank all (b=100), thresholds at the 90%, 80%, ..., 10% quantiles of the
design-pool scores, and re-rank none (b=0). A query is re-ranked iff score <= threshold (ties re-ranked). Candidates are
tested in this fixed order, from the most conservative to the most aggressive, and the scan stops at the first failure.
Loss of a query under policy t: l_q(t) = g_q * 1[q skipped], in [-1, 1] (positive = worse than re-ranking all).
Rules (applied to the mean loss over C):
  point      mean <= delta
  UCB95      mean + z_0.95 * sd / sqrt(n) <= delta                        (normal approximation, sd with ddof = 1)
  UCB-simul  same with z_(1 - 0.05/10)  (Bonferroni over the 10 non-trivial candidates)
  LTT-H      Learn-then-Test, Hoeffding p-value exp(-2 n (delta - mean)^2 / 2^2) for losses in [-1, 1], fixed-sequence
             testing at alpha = 0.05 (valid FWER control; controls P(risk of the chosen policy > delta) <= alpha)
  LTT-EB     fixed-sequence testing with the empirical-Bernstein bound (Maurer & Pontil 2009) on the loss rescaled to [0, 1]
             (variance-sensitive)
  CRC        conformal risk control on the clipped loss max(g,0)*1[skipped] in [0,1] (monotone in t):
             (n/(n+1)) * mean + 1/(n+1) <= delta; controls the EXPECTED clipped loss
Baselines on the same splits and labels:
  random-order    selector with an independent uniform score per split (realized routing), through the same rules
  random-analytic re-rank a uniformly random share b; b = smallest grid budget with (1-b) * U_g <= delta, where U_g is the
                  mean gain on C (point) or its UCB95 (UCB rules); loss = EXPECTED loss (1-b) * mean gain on T
Outputs one row per pair x selector x n x rule: mean realized share re-ranked on T, saving, mean loss on T, P(loss on T >
delta), the whole-collection diagnostic separately, and |C|, |Dp|, |T|. Pairs with N - n < 200 are skipped and listed.

--mode labels (item 10, equal total label budget n): A = WIG + UCB95 on all n labels; B = choose one of the six proxies on
n/2 design labels (largest UCB95 saving), certify it on the other n/2; C = ridge router on the six proxy percentiles trained
on n/2, certified on the other n/2; D = ridge router trained on the OTHER pairs given on the command line (their labels are
source labels, counted separately), certified on all n. All evaluated on T.

Usage: python calib46.py mag_testfull:full prime_testfull:full ... --out c46_bge
       python calib46.py ... --mode labels --ncal 200 500 --out l46_bge
"""
import argparse
import csv
import math
from statistics import NormalDist

import numpy as np

from common46 import GRID, load

Z95 = NormalDist().inv_cdf(0.95)
ZSIM = NormalDist().inv_cdf(1 - 0.05 / 10)
ALPHA = 0.05
RULES = ("point", "UCB95", "UCB-simul", "LTT-H", "LTT-EB", "CRC")
SIX = ("BM25 margin", "tag alarm", "top5 vs rest", "NQC-like spread", "WIG-like", "term coverage top1")


def candidates(design_scores):
    """thresholds from the most conservative (re-rank all) to the most aggressive (re-rank none) + their nominal budgets"""
    ts = [np.inf] + [float(np.quantile(design_scores, b / 100.0)) for b in range(90, 0, -10)] + [-np.inf]
    return ts, list(range(100, -1, -10))


def passes(loss, g_skipped_clipped, rule, delta):
    n = len(loss)
    m = loss.mean()
    if rule == "point":
        return m <= delta + 1e-12
    if rule in ("UCB95", "UCB-simul"):
        sd = loss.std(ddof=1) if n > 1 else 0.0
        return m + (Z95 if rule == "UCB95" else ZSIM) * sd / math.sqrt(n) <= delta + 1e-12
    if rule == "LTT-H":
        if m >= delta:
            return False
        return math.exp(-2 * n * (delta - m) ** 2 / 4.0) <= ALPHA
    if rule == "LTT-EB":
        u = (loss + 1) / 2
        du = (delta + 1) / 2
        v = u.var(ddof=1) if n > 1 else 0.0
        lg = math.log(2 / ALPHA)
        return u.mean() + math.sqrt(2 * v * lg / n) + 7 * lg / (3 * (n - 1)) <= du + 1e-12
    if rule == "CRC":
        return (n / (n + 1)) * g_skipped_clipped.mean() + 1.0 / (n + 1) <= delta + 1e-12
    raise ValueError(rule)


def certify(sc_C, g_C, ts, rule, delta):
    """fixed-sequence scan; returns the chosen threshold (re-rank all if even the first non-trivial candidate fails)"""
    chosen = ts[0]
    for t in ts[1:]:
        skip = sc_C > t
        loss = np.where(skip, g_C, 0.0)
        clip = np.where(skip, np.maximum(g_C, 0.0), 0.0)
        if passes(loss, clip, rule, delta):
            chosen = t
        else:
            break
    return chosen


def evaluate(sc, g, t, T):
    rer = sc[T] <= t
    lossT = float(np.where(rer, 0.0, g[T]).mean())
    lossAll = float(np.where(sc <= t, 0.0, g).mean())
    return float(rer.mean()), lossT, lossAll


def analytic_random(g_C, g, T, rule, delta):
    n = len(g_C)
    U = g_C.mean() + (0.0 if rule == "point" else (Z95 if rule != "UCB-simul" else ZSIM) * g_C.std(ddof=1) / math.sqrt(n))
    b = 100
    for bb in range(90, -1, -10):
        if (1 - bb / 100) * U <= delta + 1e-12:
            b = bb
        else:
            break
    return b / 100, float((1 - b / 100) * g[T].mean()), float((1 - b / 100) * g.mean())


def pct(x):
    r = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort")
    return r / max(1, len(x) - 1)


def ridge(X, y, alpha=1.0):
    Xb = np.column_stack([np.ones(len(X)), X])
    I = np.eye(Xb.shape[1]); I[0, 0] = 0
    w = np.linalg.solve(Xb.T @ Xb + alpha * I, Xb.T @ y)
    return lambda Z: np.column_stack([np.ones(len(Z)), Z]) @ w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+")
    ap.add_argument("--ncal", type=int, nargs="+", default=[100, 200, 500])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--mode", default="calib", choices=["calib", "labels"])
    ap.add_argument("--out", default="c46")
    a = ap.parse_args()
    delta = a.delta / 100
    data = [load(p) for p in a.pairs]
    rows, skipped = [], []
    for D in data:
        g, N = D["g"], D["n"]
        X = np.column_stack([pct(D["sig"][s]) for s in SIX])
        for n in a.ncal:
            if N - n < 200:
                skipped.append(f"{D['name']}:{D['tag']} n={n} (N={N})")
                continue
            rng = np.random.default_rng(42)
            acc = {}
            for rep in range(a.reps):
                perm = rng.permutation(N)
                C, R = perm[:n], perm[n:]
                nd = max(100, int(0.3 * len(R)))
                Dp, T = R[:nd], R[nd:]
                if a.mode == "calib":
                    sels = {"WIG": D["sig"]["WIG-like"], "tag alarm": D["sig"]["tag alarm"], "random-order": rng.random(N)}
                    for s, sc in sels.items():
                        ts, _ = candidates(sc[Dp])
                        for r in RULES:
                            t = certify(sc[C], g[C], ts, r, delta)
                            acc.setdefault((s, r), []).append(evaluate(sc, g, t, T))
                    for r in ("point", "UCB95", "UCB-simul"):
                        acc.setdefault(("random-analytic", r), []).append(analytic_random(g[C], g, T, r, delta))
                else:
                    # A: WIG with all n labels
                    sc = D["sig"]["WIG-like"]; ts, _ = candidates(sc[Dp])
                    acc.setdefault(("A: WIG + certify(n)", "UCB95"), []).append(evaluate(sc, g, certify(sc[C], g[C], ts, "UCB95", delta), T))
                    half = n // 2
                    L1, L2 = C[:half], C[half:]
                    # B: choose the proxy on L1 (largest share skipped under UCB95 on L1), certify on L2
                    best, best_skip = None, -1
                    for s in SIX:
                        sc_s = D["sig"][s]; ts_s, _ = candidates(sc_s[Dp])
                        t = certify(sc_s[L1], g[L1], ts_s, "UCB95", delta)
                        sk = float((sc_s[Dp] > t).mean())
                        if sk > best_skip:
                            best, best_skip = s, sk
                    sc = D["sig"][best]; ts, _ = candidates(sc[Dp])
                    acc.setdefault(("B: choose proxy(n/2) + certify(n/2)", "UCB95"), []).append(evaluate(sc, g, certify(sc[L2], g[L2], ts, "UCB95", delta), T))
                    # C: ridge router on L1, certify on L2 (score = -predicted gain: high predicted gain re-ranked first)
                    f = ridge(X[L1], g[L1]); sc = -f(X); ts, _ = candidates(sc[Dp])
                    acc.setdefault(("C: router(n/2) + certify(n/2)", "UCB95"), []).append(evaluate(sc, g, certify(sc[L2], g[L2], ts, "UCB95", delta), T))
                    # D: transfer router trained on the other pairs, certify on all n
                    others = [E for E in data if E is not D]
                    if others:
                        if "transfer" not in D:
                            Xo = np.vstack([np.column_stack([pct(E["sig"][s]) for s in SIX]) for E in others])
                            D["transfer"] = (ridge(Xo, np.concatenate([E["g"] for E in others])), sum(E["n"] for E in others))
                        f, nsrc = D["transfer"]
                        sc = -f(X); ts, _ = candidates(sc[Dp])
                        acc.setdefault((f"D: transfer router ({nsrc} source labels) + certify(n)", "UCB95"), []).append(
                            evaluate(sc, g, certify(sc[C], g[C], ts, "UCB95", delta), T))
            for (s, r), v in acc.items():
                v = np.array(v, float)
                rows.append({"pair": f"{D['name']}:{D['tag']}", "dataset": D["name"], "tag": D["tag"], "selector": s, "rule": r,
                             "n_cert_labels": n, "n_design_unlabelled": nd, "n_test": len(T), "N": N,
                             "share_reranked_test": round(100 * v[:, 0].mean(), 2), "saving_test": round(100 * (1 - v[:, 0].mean()), 2),
                             "mean_loss_test": round(100 * v[:, 1].mean(), 3), "p_violation_test": round(float((v[:, 1] > delta + 1e-12).mean()), 3),
                             "p_violation_whole_collection": round(float((v[:, 2] > delta + 1e-12).mean()), 3), "splits": a.reps})
    with open(a.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(a.out + "_skipped.txt", "w") as fh:
        fh.write("pairs skipped because N - n < 200:\n" + "\n".join(skipped) + "\n")
    print(f"saved {a.out}.csv ({len(rows)} rows); skipped: {skipped}")


if __name__ == "__main__":
    main()
