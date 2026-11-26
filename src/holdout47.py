#!/usr/bin/env python3
"""
holdout47.py: test the practitioner checklist on collections that played NO part in building it.

The checklist (paper, Figure "Practitioner checklist"), frozen here as an algorithm before the held-out runs:
  step 1  label n = 200 queries (certification sample C), estimate the mean gain g_C
  step 2  regime from rho = g_C / delta:   rho <= LOW (1)   -> LOW:  re-rank a random share (analytic random, UCB95)
                                            rho >= HIGH (15) -> HIGH: re-rank every query
                                            otherwise        -> MIDDLE: label-free selector (WIG), steps 3-4
  step 3  WIG threshold certified with UCB95 on C (design thresholds from an unlabelled pool Dp, as calib46.py)
  step 4  compare with analytic random (UCB95) on the same labels: keep WIG only if its ESTIMATED saving (share of the
          unlabelled pool Dp it skips) is larger than random's saving 1 - b; otherwise use random
  (steps 5-6, selector cost and query groups, are not automated: WIG's own cost is < 3 ms, and groups need the application)

Per split (default 200, seed 42), with the same C / Dp / T split as calib46.py:
  candidate policies, all certified on C with UCB95 at tolerance delta:
     all       re-rank every query
     random    analytic random, b = smallest grid budget with (1 - b) * UCB95(g_C) <= delta; loss = (1 - b) * mean g_T
     <proxy>   each of the six label-free proxies + tag alarm, threshold certified with UCB95
  hindsight best = largest saving on T among candidates whose loss on T is <= delta (uses T labels: an upper bound,
                   not a policy)
  checklist     = the policy the frozen checklist picks from C and Dp only
  ablations     = always WIG; always random; "no regime step" (step 4 only: WIG vs random by estimated saving)
Outcome per split: saving on T, violation (loss on T > delta), regret = hindsight saving - saving on non-violating
splits (percentage points of re-ranking), and "correct" = no violation and regret <= 5 points (half a grid step).
The regime is also computed from the whole collection (rho_all = mean gain / delta) to measure how often 200 labels
call the regime that all labels would call.
Boundaries in units of delta: everything is run for delta in {0.5, 1, 2} points and for MRR@10 and nDCG@10, so the
1 and 15 boundaries are tested as 1*delta and 15*delta, not as fixed MRR points.

Usage:
  python holdout47.py beir_nq:h47bge beir_hotpotqa:h47bge ... --insample mag_testfull:full ... --out h47
Outputs: h47.md (readable), h47_pairs.csv (one row per pair x metric x delta), h47_buckets.csv (by rho bucket).
"""
import argparse
import csv
import math
import os
from collections import Counter

import numpy as np

from calib46 import SIX, analytic_random, candidates, certify, evaluate
from common46 import load

LOW, HIGH = 1.0, 15.0           # checklist boundaries, in units of delta (frozen)
REGRET_OK = 5.0                 # "correct" = no violation and regret <= 5 percentage points (frozen)
RULE = "UCB95"
BUCKETS = [(-np.inf, 1), (1, 3), (3, 7), (7, 15), (15, 30), (30, np.inf)]


def regime(rho):
    return "low" if rho <= LOW else ("high" if rho >= HIGH else "middle")


def family(name):
    return "all" if name == "all" else ("random" if name == "random" else "selector")


def run_pair(D, n, reps, delta, seed=42):
    g, N = D["g"], D["n"]
    rng = np.random.default_rng(seed)
    sels = [s for s in SIX if s in D["sig"]]
    out = []
    for _ in range(reps):
        perm = rng.permutation(N)
        C, R = perm[:n], perm[n:]
        nd = max(100, int(0.3 * len(R)))
        Dp, T = R[:nd], R[nd:]
        pol = {"all": {"save": 0.0, "loss": 0.0, "est": 0.0}}
        b, lossT, _ = analytic_random(g[C], g, T, RULE, delta)
        pol["random"] = {"save": 100 * (1 - b), "loss": lossT, "est": 100 * (1 - b)}
        for s in sels:
            sc = D["sig"][s]
            ts, _ = candidates(sc[Dp])
            t = certify(sc[C], g[C], ts, RULE, delta)
            share, lT, _ = evaluate(sc, g, t, T)
            pol[s] = {"save": 100 * (1 - share), "loss": lT, "est": 100 * float((sc[Dp] > t).mean())}
        feas = {k: v["save"] for k, v in pol.items() if v["loss"] <= delta + 1e-12}
        best = max(feas, key=lambda k: (feas[k], k == "all"))  # ties: prefer the simpler policy only on exact ties
        best_save = feas[best]
        rho_c = float(g[C].mean()) / delta
        reg = regime(rho_c)
        if reg == "low":
            choice = "random"
        elif reg == "high":
            choice = "all"
        else:
            choice = "WIG-like" if pol["WIG-like"]["est"] > pol["random"]["est"] else "random"
        noreg = "WIG-like" if pol["WIG-like"]["est"] > pol["random"]["est"] else "random"

        def outcome(k):
            v = pol[k]
            viol = v["loss"] > delta + 1e-12
            regret = float("nan") if viol else best_save - v["save"]   # regret is measured on non-violating splits
            return {"policy": k, "save": v["save"], "viol": viol, "regret": regret,
                    "correct": (not viol) and regret <= REGRET_OK + 1e-9, "agree": family(k) == family(best)}

        out.append({"regime_C": reg, "rho_C": rho_c, "best": best, "best_save": best_save,
                    "checklist": outcome(choice), "no_regime": outcome(noreg),
                    "always_wig": outcome("WIG-like"), "always_random": outcome("random"),
                    "always_all": outcome("all")})
    return out


def summarize(res, key):
    xs = [r[key] for r in res]
    return {"save": float(np.mean([x["save"] for x in xs])), "viol": float(np.mean([x["viol"] for x in xs])),
            "regret": float(np.nanmean([x["regret"] for x in xs])) if any(not x["viol"] for x in xs) else float("nan"), "correct": float(np.mean([x["correct"] for x in xs])),
            "agree": float(np.mean([x["agree"] for x in xs]))}


STRATS = ("checklist", "no_regime", "always_wig", "always_random", "always_all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", help="held-out folder:tag pairs")
    ap.add_argument("--insample", nargs="*", default=[], help="the pairs the checklist was built from (comparison)")
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.5, 1.0, 2.0], help="tolerance, in metric points")
    ap.add_argument("--metrics", nargs="+", default=["MRR", "ndcg"])
    ap.add_argument("--ncal", type=int, default=200)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", default="h47")
    a = ap.parse_args()

    rows, skipped = [], []
    for group, pairs in (("held-out", a.pairs), ("in-sample", a.insample)):
        for p in pairs:
            folder, tag = p.split(":")
            if not os.path.exists(os.path.join(folder, f"selective_pq_{tag}.jsonl")):
                skipped.append(f"{p}: no per-query file")
                continue
            for metric in a.metrics:
                try:
                    D = load(p, metric)
                except (KeyError, TypeError):
                    skipped.append(f"{p}: no {metric}")
                    continue
                if D["n"] - a.ncal < 200:
                    skipped.append(f"{p}: N = {D['n']} too small for n = {a.ncal}")
                    continue
                for dlt in a.deltas:
                    delta = dlt / 100
                    res = run_pair(D, a.ncal, a.reps, delta)
                    rho_all = float(D["g"].mean()) / delta
                    row = {"group": group, "pair": p, "metric": metric, "delta_points": dlt, "N": D["n"],
                           "mean_gain_points": round(100 * float(D["g"].mean()), 2), "rho_all": round(rho_all, 2),
                           "regime_all": regime(rho_all)}
                    cnt = Counter(r["regime_C"] for r in res)
                    for k in ("low", "middle", "high"):
                        row[f"regime_C_{k}"] = round(cnt[k] / len(res), 3)
                    row["regime_C_agrees_all"] = round(cnt[regime(rho_all)] / len(res), 3)
                    row["hindsight_save"] = round(float(np.mean([r["best_save"] for r in res])), 2)
                    fam = Counter(family(r["best"]) for r in res)
                    row["hindsight_family"] = fam.most_common(1)[0][0]
                    for k in ("all", "random", "selector"):
                        row[f"hindsight_{k}"] = round(fam[k] / len(res), 3)
                    row["hindsight_top_selector"] = Counter(r["best"] for r in res if family(r["best"]) == "selector").most_common(1)[0][0] \
                        if fam["selector"] else ""
                    for s in STRATS:
                        sm = summarize(res, s)
                        for k, v in sm.items():
                            row[f"{s}_{k}"] = round(v, 3 if k in ("viol", "correct", "agree") else 2)
                    rows.append(row)
                    print(f"{group:9s} {p:32s} {metric:4s} d={dlt:3.1f} rho={rho_all:6.1f} {row['regime_all']:6s} "
                          f"| checklist save {row['checklist_save']:5.1f} viol {row['checklist_viol']:.3f} "
                          f"regret {row['checklist_regret']:5.1f} correct {row['checklist_correct']:.2f} "
                          f"| hindsight {row['hindsight_save']:5.1f} ({row['hindsight_family']})", flush=True)

    if not rows:
        raise SystemExit("no pairs evaluated: " + "; ".join(skipped))
    with open(a.out + "_pairs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- by rho bucket (tests the boundaries in units of delta)
    brows = []
    for group in ("held-out", "in-sample"):
        for lo, hi in BUCKETS:
            sub = [r for r in rows if r["group"] == group and lo < r["rho_all"] <= hi]
            if not sub:
                continue
            brows.append({"group": group, "rho_bucket": f"({lo:g}, {hi:g}]", "cells": len(sub),
                          "hindsight_save": round(np.mean([r["hindsight_save"] for r in sub]), 2),
                          "hindsight_all": round(np.mean([r["hindsight_all"] for r in sub]), 3),
                          "hindsight_random": round(np.mean([r["hindsight_random"] for r in sub]), 3),
                          "hindsight_selector": round(np.mean([r["hindsight_selector"] for r in sub]), 3),
                          "random_save": round(np.mean([r["always_random_save"] for r in sub]), 2),
                          "wig_save": round(np.mean([r["always_wig_save"] for r in sub]), 2),
                          "wig_viol": round(np.mean([r["always_wig_viol"] for r in sub]), 3),
                          "checklist_save": round(np.mean([r["checklist_save"] for r in sub]), 2),
                          "checklist_correct": round(np.mean([r["checklist_correct"] for r in sub]), 3)})
    with open(a.out + "_buckets.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(brows[0]))
        w.writeheader()
        w.writerows(brows)

    # ---- readable report
    md = [f"# holdout47: practitioner checklist on held-out collections ({a.reps} splits, n = {a.ncal} labels, rule {RULE}, "
          f"boundaries {LOW:g}*delta and {HIGH:g}*delta, correct = no violation and regret <= {REGRET_OK:g} points)\n"]
    for group in ("held-out", "in-sample"):
        sub = [r for r in rows if r["group"] == group]
        if not sub:
            continue
        md.append(f"## {group}: {len({r['pair'] for r in sub})} pairs, {len(sub)} pair x metric x delta cells\n")
        md.append("strategy        | mean saving | P(violation) | mean regret | P(correct) | P(same family as hindsight)")
        md.append("----------------|------------:|-------------:|------------:|-----------:|---------------------------:")
        for s in STRATS:
            md.append(f"{s:15s} | {np.mean([r[s + '_save'] for r in sub]):11.1f} | {np.mean([r[s + '_viol'] for r in sub]):12.3f} | "
                      f"{np.nanmean([r[s + '_regret'] for r in sub]):11.1f} | {np.mean([r[s + '_correct'] for r in sub]):10.2f} | "
                      f"{np.mean([r[s + '_agree'] for r in sub]):10.2f}")
        md.append(f"hindsight best  | {np.mean([r['hindsight_save'] for r in sub]):11.1f} |   (uses test labels: upper bound)\n")
        md.append(f"regime called from {a.ncal} labels = regime from all labels: "
                  f"{np.mean([r['regime_C_agrees_all'] for r in sub]):.2f} of splits\n")
        md.append("pair                             metric delta  gain  rho  regime | C: low/mid/high | checklist save viol regret correct | hindsight save family (top selector)")
        for r in sub:
            md.append(f"{r['pair']:32s} {r['metric']:5s} {r['delta_points']:4.1f} {r['mean_gain_points']:6.1f} {r['rho_all']:5.1f} {r['regime_all']:6s} | "
                      f"{r['regime_C_low']:.2f}/{r['regime_C_middle']:.2f}/{r['regime_C_high']:.2f} | "
                      f"{r['checklist_save']:5.1f} {r['checklist_viol']:.3f} {r['checklist_regret']:5.1f} {r['checklist_correct']:.2f} | "
                      f"{r['hindsight_save']:5.1f} {r['hindsight_family']} ({r['hindsight_top_selector']})")
        md.append("")
    md.append("## by rho = mean gain / delta (whole collection)\n")
    md.append("group     rho bucket    cells | hindsight save  P(all/random/selector best) | random save | WIG save viol | checklist save correct")
    for b in brows:
        md.append(f"{b['group']:9s} {b['rho_bucket']:13s} {b['cells']:5d} | {b['hindsight_save']:14.1f}  "
                  f"{b['hindsight_all']:.2f}/{b['hindsight_random']:.2f}/{b['hindsight_selector']:.2f}               | "
                  f"{b['random_save']:11.1f} | {b['wig_save']:8.1f} {b['wig_viol']:.3f} | {b['checklist_save']:14.1f} {b['checklist_correct']:.2f}")
    md.append("\nskipped: " + ("; ".join(skipped) if skipped else "none"))
    open(a.out + ".md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
