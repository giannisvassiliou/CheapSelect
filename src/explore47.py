#!/usr/bin/env python3
"""
explore47.py: EXPLORATORY (not pre-registered) follow-up to holdout47.py, written after seeing the held-out results.
Question: does using the same 200 labels to CHOOSE the selector (not only its threshold) close the gap to the best
policy in hindsight, e.g. find the BM25 margin on HotpotQA, without breaking the tolerance?

Same splits as holdout47.py (seed 42: C = 200 labelled, Dp = unlabelled design pool, T = untouched test), same
candidates (all, analytic random, six proxies + tag alarm, each certified with a confidence bound), same hindsight best.
Strategies (all decided from C labels and Dp scores only):
  checklist        the frozen rule of holdout47.py (WIG; rho <= 1 random; rho >= 15 all)
  margin-fixed     the same rule with the BM25 margin in place of WIG   (informed by the held-out results: hindsight)
  choose-split     choose among random and the proxies on C1 (first half of C), certify the chosen one on C2;
                   rho <= 1 -> random; no rule for high rho
  choose-simul     certify every proxy on all of C with a Bonferroni bound over proxies x thresholds
                   (z at 1 - 0.05 / (7 * 10)), keep the one with the largest estimated saving (or random);
                   rho <= 1 -> random; no rule for high rho
  choose-simul+hi  choose-simul, plus the checklist's rule rho >= 15 -> re-rank all
  choose-naive     certify every proxy with UCB95 on C and keep the largest estimated saving (double use of C:
                   shows what ignoring the multiplicity costs in violations)
Usage: python explore47.py <held-out pairs> --insample <in-sample pairs> --out e47
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
STRATS = ("checklist", "margin-fixed", "choose-split", "choose-simul", "choose-simul+hi", "choose-naive")


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


def random_z(g_C, z, delta):
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

        def proxy_pol(s, cert_idx, z):
            sc = D["sig"][s]
            t = certify_z(sc[cert_idx], g[cert_idx], TS[s], z, delta)
            share, lT, _ = evaluate(sc, g, t, T)
            return {"save": 100 * (1 - share), "loss": lT, "est": 100 * float((sc[Dp] > t).mean())}

        def rand_pol(idx, z):
            b = random_z(g[idx], z, delta)
            return {"save": 100 * (1 - b), "loss": (1 - b) * gT, "est": 100 * (1 - b)}

        allp = {"save": 0.0, "loss": 0.0, "est": 0.0}
        # candidates and hindsight best: identical to holdout47.py (UCB95 on all of C)
        cand = {"all": allp, "random": rand_pol(C, Z95)}
        for s in sels:
            cand[s] = proxy_pol(s, C, Z95)
        feas = {k: v["save"] for k, v in cand.items() if v["loss"] <= delta + 1e-12}
        best_save = max(feas.values())
        rho = float(g[C].mean()) / delta
        reg = "low" if rho <= LOW else ("high" if rho >= HIGH else "middle")

        def pick(pols):
            k = max(pols, key=lambda k: pols[k]["est"])
            return pols[k]

        S = {}
        # frozen checklist and the margin variant
        for name, prox in (("checklist", "WIG-like"), ("margin-fixed", "BM25 margin")):
            if reg == "low":
                S[name] = cand["random"]
            elif reg == "high":
                S[name] = allp
            else:
                S[name] = cand[prox] if cand[prox]["est"] > cand["random"]["est"] else cand["random"]
        # choose-split: choose on C1 (UCB95), certify the chosen one on C2 (UCB95)
        if reg == "low":
            S["choose-split"] = rand_pol(C, Z95)
        else:
            est1 = {s: proxy_pol(s, C1, Z95)["est"] for s in sels}
            est1["random"] = 100 * (1 - random_z(g[C1], Z95, delta))
            k = max(est1, key=est1.get)
            S["choose-split"] = rand_pol(C2, Z95) if k == "random" else proxy_pol(k, C2, Z95)
        # choose-simul: Bonferroni over proxies x thresholds, all of C
        sim = {s: proxy_pol(s, C, zsim) for s in sels}
        sim["random"] = rand_pol(C, zsim)
        S["choose-simul"] = cand["random"] if reg == "low" else pick(sim)
        S["choose-simul+hi"] = allp if reg == "high" else S["choose-simul"]
        # choose-naive: UCB95 for each, keep the largest estimated saving (double use of C)
        S["choose-naive"] = cand["random"] if reg == "low" else pick({k: v for k, v in cand.items() if k != "all"})
        rec = {"best_save": best_save}
        for k, v in S.items():
            viol = v["loss"] > delta + 1e-12
            regret = float("nan") if viol else best_save - v["save"]
            rec[k] = {"save": v["save"], "viol": viol, "regret": regret, "correct": (not viol) and regret <= REGRET_OK + 1e-9}
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+")
    ap.add_argument("--insample", nargs="*", default=[])
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--metrics", nargs="+", default=["MRR", "ndcg"])
    ap.add_argument("--ncal", type=int, default=200)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", default="e47")
    a = ap.parse_args()
    rows = []
    for group, pairs in (("held-out", a.pairs), ("in-sample", a.insample)):
        for p in pairs:
            folder, tag = p.split(":")
            if not os.path.exists(os.path.join(folder, f"selective_pq_{tag}.jsonl")):
                continue
            for metric in a.metrics:
                try:
                    D = load(p, metric)
                except (KeyError, TypeError):
                    continue
                if D["n"] - a.ncal < 200:
                    continue
                for dlt in a.deltas:
                    res = run_pair(D, a.ncal, a.reps, dlt / 100)
                    row = {"group": group, "pair": p, "metric": metric, "delta_points": dlt,
                           "rho_all": round(float(D["g"].mean()) / (dlt / 100), 2),
                           "hindsight_save": round(float(np.mean([r["best_save"] for r in res])), 2)}
                    for s in STRATS:
                        xs = [r[s] for r in res]
                        row[f"{s}_save"] = round(float(np.mean([x["save"] for x in xs])), 2)
                        row[f"{s}_viol"] = round(float(np.mean([x["viol"] for x in xs])), 3)
                        rg = [x["regret"] for x in xs if not x["viol"]]
                        row[f"{s}_regret"] = round(float(np.mean(rg)), 2) if rg else float("nan")
                        row[f"{s}_correct"] = round(float(np.mean([x["correct"] for x in xs])), 3)
                    rows.append(row)
                    print(f"{group:9s} {p:30s} {metric:4s} d={dlt:3.1f} rho={row['rho_all']:6.1f} hindsight {row['hindsight_save']:5.1f} | "
                          + " | ".join(f"{s} {row[s + '_save']:5.1f}/{row[s + '_viol']:.2f}" for s in STRATS), flush=True)
    with open(a.out + "_pairs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = ["# explore47 (EXPLORATORY, not pre-registered): choosing the selector with the same labels\n"]
    for group in ("held-out", "in-sample"):
        sub = [r for r in rows if r["group"] == group]
        if not sub:
            continue
        md.append(f"## {group}: {len({r['pair'] for r in sub})} pairs, {len(sub)} cells; hindsight best saves "
                  f"{np.mean([r['hindsight_save'] for r in sub]):.1f}\n")
        md.append("strategy         | mean saving | P(violation) | mean regret | P(correct)")
        md.append("-----------------|------------:|-------------:|------------:|-----------:")
        for s in STRATS:
            md.append(f"{s:16s} | {np.mean([r[s + '_save'] for r in sub]):11.1f} | {np.mean([r[s + '_viol'] for r in sub]):12.3f} | "
                      f"{np.nanmean([r[s + '_regret'] for r in sub]):11.1f} | {np.mean([r[s + '_correct'] for r in sub]):10.2f}")
        md.append("")
        md.append("per pair at MRR, delta = 1:  saving / P(violation)")
        md.append(f"{'pair':30s} {'rho':>5s} {'hind':>5s} | " + " | ".join(f"{s:>15s}" for s in STRATS))
        for r in sub:
            if r["metric"] == "MRR" and r["delta_points"] == 1.0:
                md.append(f"{r['pair']:30s} {r['rho_all']:5.1f} {r['hindsight_save']:5.1f} | "
                          + " | ".join(f"{r[s + '_save']:7.1f} / {r[s + '_viol']:.2f}" for s in STRATS))
        md.append("")
    open(a.out + ".md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
