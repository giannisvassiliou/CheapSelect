#!/usr/bin/env python3
"""
holdout48.py: SECOND pre-registered held-out test, of the revised checklist (v2).

v2 was designed after the first held-out test (holdout47.py / explore47.py), where the fixed choice of WIG left about
10 points of saving and the rule "rho >= 15 -> re-rank all" missed HotpotQA. v2 (frozen here, before any run on the
collections of this test):
  1  label n = 200 queries (C); unlabelled design pool Dp gives the candidate thresholds (as calib46.py)
  2  certify EVERY cheap selector (six proxies + tag alarm) and analytic random on all of C with a one-sided normal
     bound corrected for multiplicity, z at 1 - 0.05 / (number of selectors * 10 thresholds)
  3  keep the one with the largest estimated saving (share of Dp skipped; 1 - b for random); if none skips anything,
     re-rank every query
  4  rho = mean gain on C / delta is reported, but is NOT a decision rule
Compared on the same splits (seed 42, 200 splits) with: v1 (the frozen checklist of holdout47.py), choose-split
(choose on C1, certify on C2), always WIG, always random, always re-rank all, and the best certified policy in hindsight
(an upper bound that uses test labels). Outcomes on untouched test queries T as in holdout47.py.
Usage: python holdout48.py <new pairs> --ref47 <first held-out pairs> --insample <27 pairs> --out h48
"""
import argparse
import csv
import math
import os
from statistics import NormalDist

import numpy as np

from calib46 import SIX, Z95, analytic_random, candidates, evaluate
from common46 import load

LOW, HIGH, REGRET_OK = 1.0, 15.0, 5.0
STRATS = ("v2", "v1", "choose_split", "always_wig", "always_random", "always_all")


def certify_z(sc_C, g_C, ts, z, delta):
    chosen = ts[0]
    for t in ts[1:]:
        loss = np.where(sc_C > t, g_C, 0.0)
        sd = loss.std(ddof=1) if len(loss) > 1 else 0.0
        if loss.mean() + z * sd / math.sqrt(len(loss)) <= delta + 1e-12:
            chosen = t
        else:
            break
    return chosen


def random_b(g_C, z, delta):
    U = g_C.mean() + z * g_C.std(ddof=1) / math.sqrt(len(g_C))
    b = 100
    for bb in range(90, -1, -10):
        if (1 - bb / 100) * U <= delta + 1e-12:
            b = bb
        else:
            break
    return b / 100


def run_pair(D, n, reps, delta, seed=42):
    g, N = D["g"], D["n"]
    sels = [s for s in SIX if s in D["sig"]]
    zsim = NormalDist().inv_cdf(1 - 0.05 / (len(sels) * 10))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        perm = rng.permutation(N)
        C, R = perm[:n], perm[n:]
        nd = max(100, int(0.3 * len(R)))
        Dp, T = R[:nd], R[nd:]
        C1, C2 = C[: n // 2], C[n // 2:]
        TS = {s: candidates(D["sig"][s][Dp])[0] for s in sels}
        gT = float(g[T].mean())

        def prox(s, idx, z):
            sc = D["sig"][s]
            t = certify_z(sc[idx], g[idx], TS[s], z, delta)
            share, lT, _ = evaluate(sc, g, t, T)
            return {"name": s, "save": 100 * (1 - share), "loss": lT, "est": 100 * float((sc[Dp] > t).mean())}

        def rand(idx, z):
            b = random_b(g[idx], z, delta)
            return {"name": "random", "save": 100 * (1 - b), "loss": (1 - b) * gT, "est": 100 * (1 - b)}

        allp = {"name": "all", "save": 0.0, "loss": 0.0, "est": 0.0}
        cand = {"all": allp, "random": rand(C, Z95)}
        for s in sels:
            cand[s] = prox(s, C, Z95)
        feas = [v["save"] for v in cand.values() if v["loss"] <= delta + 1e-12]
        best_save = max(feas)
        rho = float(g[C].mean()) / delta
        S = {}
        # v2
        sim = [prox(s, C, zsim) for s in sels] + [rand(C, zsim)]
        top = max(sim, key=lambda v: v["est"])
        S["v2"] = top if top["est"] > 0 else allp
        # v1 (frozen checklist of holdout47.py)
        if rho <= LOW:
            S["v1"] = cand["random"]
        elif rho >= HIGH:
            S["v1"] = allp
        else:
            S["v1"] = cand["WIG-like"] if cand["WIG-like"]["est"] > cand["random"]["est"] else cand["random"]
        # choose on C1, certify on C2
        e1 = [prox(s, C1, Z95) for s in sels] + [rand(C1, Z95)]
        k = max(e1, key=lambda v: v["est"])["name"]
        S["choose_split"] = rand(C2, Z95) if k == "random" else prox(k, C2, Z95)
        S["always_wig"] = cand["WIG-like"]
        S["always_random"] = cand["random"]
        S["always_all"] = allp
        rec = {"best": best_save, "rho": rho, "v2_name": S["v2"]["name"]}
        for k_, v in S.items():
            viol = v["loss"] > delta + 1e-12
            rg = float("nan") if viol else best_save - v["save"]
            rec[k_] = {"save": v["save"], "viol": viol, "regret": rg, "correct": (not viol) and rg <= REGRET_OK + 1e-9}
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", help="pairs of THIS test (second held-out set)")
    ap.add_argument("--ref47", nargs="*", default=[], help="first held-out set (comparison only)")
    ap.add_argument("--insample", nargs="*", default=[], help="the 27 pairs of the study (comparison only)")
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--metrics", nargs="+", default=["MRR", "ndcg"])
    ap.add_argument("--ncal", type=int, default=200)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", default="h48")
    a = ap.parse_args()
    rows, skipped = [], []
    for group, pairs in (("held-out-2", a.pairs), ("held-out-1", a.ref47), ("in-sample", a.insample)):
        for p in pairs:
            folder, tag = p.split(":")
            if not os.path.exists(os.path.join(folder, f"selective_pq_{tag}.jsonl")):
                skipped.append(f"{p}: missing")
                continue
            for metric in a.metrics:
                try:
                    D = load(p, metric)
                except (KeyError, TypeError):
                    continue
                if D["n"] - a.ncal < 200:
                    skipped.append(f"{p}: N = {D['n']}")
                    continue
                for dlt in a.deltas:
                    res = run_pair(D, a.ncal, a.reps, dlt / 100)
                    row = {"group": group, "pair": p, "metric": metric, "delta_points": dlt, "N": D["n"],
                           "rho_all": round(float(D["g"].mean()) / (dlt / 100), 2),
                           "hindsight_save": round(float(np.mean([r["best"] for r in res])), 2)}
                    names = [r["v2_name"] for r in res]
                    row["v2_top_choice"] = max(set(names), key=names.count)
                    row["v2_top_choice_share"] = round(names.count(row["v2_top_choice"]) / len(names), 2)
                    for s in STRATS:
                        xs = [r[s] for r in res]
                        rg = [x["regret"] for x in xs if not x["viol"]]
                        row[f"{s}_save"] = round(float(np.mean([x["save"] for x in xs])), 2)
                        row[f"{s}_viol"] = round(float(np.mean([x["viol"] for x in xs])), 3)
                        row[f"{s}_regret"] = round(float(np.mean(rg)), 2) if rg else float("nan")
                        row[f"{s}_correct"] = round(float(np.mean([x["correct"] for x in xs])), 3)
                    rows.append(row)
                    print(f"{group:10s} {p:30s} {metric:4s} d={dlt:3.1f} rho={row['rho_all']:6.1f} | v2 {row['v2_save']:5.1f}/{row['v2_viol']:.3f} "
                          f"({row['v2_top_choice']}) | v1 {row['v1_save']:5.1f}/{row['v1_viol']:.3f} | hindsight {row['hindsight_save']:5.1f}", flush=True)
    with open(a.out + "_pairs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = [f"# holdout48: revised checklist (v2) on a second held-out set ({a.reps} splits, n = {a.ncal})\n"]
    for group in ("held-out-2", "held-out-1", "in-sample"):
        sub = [r for r in rows if r["group"] == group]
        if not sub:
            continue
        tag = " (THE TEST)" if group == "held-out-2" else " (comparison only)"
        md.append(f"## {group}{tag}: {len({r['pair'] for r in sub})} pairs, {len(sub)} cells; hindsight best saves "
                  f"{np.mean([r['hindsight_save'] for r in sub]):.1f}\n")
        md.append("strategy       | mean saving | P(violation) | mean regret | P(correct)")
        md.append("---------------|------------:|-------------:|------------:|-----------:")
        for s in STRATS:
            md.append(f"{s:14s} | {np.mean([r[s + '_save'] for r in sub]):11.1f} | {np.mean([r[s + '_viol'] for r in sub]):12.3f} | "
                      f"{np.nanmean([r[s + '_regret'] for r in sub]):11.1f} | {np.mean([r[s + '_correct'] for r in sub]):10.2f}")
        d = np.array([r["v2_save"] - r["v1_save"] for r in sub])
        md.append(f"\nv2 - v1 saving per cell: mean {d.mean():+.1f}, v2 higher in {int((d > 0.05).sum())}, equal in "
                  f"{int((np.abs(d) <= 0.05).sum())}, lower in {int((d < -0.05).sum())} of {len(d)} cells\n")
        md.append("per pair at MRR, delta = 1: saving / P(violation)")
        md.append(f"{'pair':30s} {'rho':>5s} {'hind':>5s} | {'v2':>13s} {'(chosen)':18s} | {'v1':>13s} | {'choose_split':>13s}")
        for r in sub:
            if r["metric"] == "MRR" and r["delta_points"] == 1.0:
                md.append(f"{r['pair']:30s} {r['rho_all']:5.1f} {r['hindsight_save']:5.1f} | {r['v2_save']:5.1f} / {r['v2_viol']:.3f} "
                          f"({r['v2_top_choice'][:12]} {r['v2_top_choice_share']:.2f}) | {r['v1_save']:5.1f} / {r['v1_viol']:.3f} | "
                          f"{r['choose_split_save']:5.1f} / {r['choose_split_viol']:.3f}")
        md.append("")
    md.append("skipped: " + ("; ".join(skipped) if skipped else "none"))
    open(a.out + ".md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
