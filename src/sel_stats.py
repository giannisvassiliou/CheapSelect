#!/usr/bin/env python3
"""
sel_stats.py: significance tests for selective reranking (frozen test run).

Reads <dataset>/selective_pq_<tag>.jsonl written by combo_search33.py --selective
and tests, per dataset and pooled over all datasets:
  A. tag alarm (its own firing rate)  vs  random selection of the same size
  B. tag alarm  vs  small BM25 margin   at the same budget  (paired sign-flip test)
  C. tag alarm  vs  reranker on ALL     (is the loss significant? + 95% CI)
  D. probe top-3 (keep BM25 if agree) vs margin at the same cost
  E. tag alarm vs NQC, WIG, top5-vs-rest, term coverage (standard label-free signals)
  G. every selector at the tag alarm's budget vs reranking all
  H. full budget curves 10-100% per selector, CSV for the figure
  I. non-inferiority vs rerank-all (margin --ni-margins, 95% CI)
  J. held-out budget calibration on n labelled queries
  F. per query type (if the data has qtype, e.g. DBpedia-Entity)
Selection rules are exactly those of combo_search31/32 (nothing is tuned here).

Usage:
  python sel_stats.py amazon_paper mag_paper prime_full --tag final
"""
import argparse
import json
import os

import numpy as np

KEYS = ("Hit@1", "MRR")
BUDGETS = (30, 40, 50)


def load(folder, tag):
    fn = os.path.join(folder, f"selective_pq_{tag}.jsonl" if tag else "selective_pq.jsonl")
    rows = [json.loads(l) for l in open(fn)]
    bm = {k: np.array([r["bm"][k] for r in rows]) for k in KEYS}
    ce = {k: np.array([r["ce"][k] for r in rows]) for k in KEYS}
    alarm = np.array([r["alarm"] for r in rows])
    margin = np.array([r["margin"] for r in rows])
    agree3 = np.array([r["probe3_agree"] for r in rows])
    qpp = {}
    for name in (rows[0].get("qpp") or {}):
        qpp[name] = np.array([r["qpp"][name] for r in rows])
    return bm, ce, alarm, margin, agree3, qpp


def load_qtypes(folder, tag):
    fn = os.path.join(folder, f"selective_pq_{tag}.jsonl" if tag else "selective_pq.jsonl")
    return np.array([json.loads(l).get("qtype", "") or "" for l in open(fn)])


def pick(order, k, n):
    s = np.zeros(n, bool)
    s[order[:k]] = True
    return s


def per_q(sel, bm, ce, k):
    return np.where(sel, ce[k], bm[k])


def signflip(d, n=20000, seed=0):
    d = np.asarray(d, float)
    if not np.any(d):
        return 1.0
    rng = np.random.default_rng(seed)
    null = np.abs((rng.choice([-1.0, 1.0], size=(n, len(d))) * d).mean(axis=1))
    return float((np.sum(null >= abs(d.mean()) - 1e-12) + 1) / (n + 1))


def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = np.asarray(d)[idx].mean(axis=1)
    return 100 * np.percentile(m, 2.5), 100 * np.percentile(m, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--tag", default="final")
    ap.add_argument("--out", default="sel_stats.md")
    ap.add_argument("--ni-margins", type=float, nargs="+", default=[1.0, 0.5],
                    help="non-inferiority margins in MRR points (fixed before looking at results); first one used in J")
    ap.add_argument("--cal-sizes", type=int, nargs="+", default=[50, 100, 200])
    ap.add_argument("--cal-reps", type=int, default=200)
    args = ap.parse_args()
    out = []

    def p(s=""):
        print(s)
        out.append(s)

    pooled = {"A_tag": {k: [] for k in KEYS}, "A_rand": {k: [] for k in KEYS}}
    pooledB = {b: {k: [] for k in KEYS} for b in list(BUDGETS) + ["nat"]}
    pooledC = {k: [] for k in KEYS}
    pooledD = {k: [] for k in KEYS}
    pooledE = {}
    curve_rows = []
    rng = np.random.default_rng(0)

    for ds in args.datasets:
        name = os.path.basename(ds.rstrip("/"))
        bm, ce, alarm, margin, agree3, qpp = load(ds, args.tag)
        n = len(margin)
        o_tag = np.lexsort((margin, -alarm))
        o_mar = np.argsort(margin, kind="mergesort")
        fired = alarm > 0
        kf = int(fired.sum())
        p(f"## {name}  ({n} questions; tag alarm fires on {kf} = {100 * kf / n:.0f}%)\n")

        # A. tag alarm vs random of same size: permutation p = share of random draws >= tag alarm
        p("A. tag alarm vs RANDOM selection of the same size (10,000 random draws)")
        for k in KEYS:
            t = per_q(fired, bm, ce, k).mean()
            gain = ce[k] - bm[k]
            draws = np.array([gain[rng.permutation(n)[:kf]].sum() for _ in range(10000)]) / n + bm[k].mean()
            pv = (np.sum(draws >= t - 1e-12) + 1) / (len(draws) + 1)
            p(f"   {k:5s}: tag {100 * t:5.1f}  random {100 * draws.mean():5.1f}  diff {100 * (t - draws.mean()):+5.1f}  p={pv:.4f}")
            pooled["A_tag"][k].append(per_q(fired, bm, ce, k))
            pooled["A_rand"][k].append(gain)  # stored for pooled draw below
        p()

        # B. tag alarm vs margin at matched budgets (paired)
        p("B. tag alarm vs SMALL BM25 MARGIN, same number of questions reranked (paired sign-flip)")
        p(f"   {'budget':8s} " + "  ".join(f"{k + ' diff (p)':>20s}" for k in KEYS))
        for b in list(BUDGETS) + ["nat"]:
            kk = kf if b == "nat" else int(round(n * b / 100))
            st, sm = pick(o_tag, kk, n), pick(o_mar, kk, n)
            cells = []
            for k in KEYS:
                d = per_q(st, bm, ce, k) - per_q(sm, bm, ce, k)
                pooledB[b][k].append(d)
                cells.append(f"{100 * d.mean():+5.1f} (p={signflip(d):.3f})")
            lab = f"{100 * kf / n:.0f}% nat" if b == "nat" else f"{b}%"
            p(f"   {lab:8s} " + "  ".join(f"{c:>20s}" for c in cells))
        p()

        # C. tag alarm vs rerank everything
        p("C. tag alarm vs reranker on ALL questions (negative = tag alarm loses)")
        for k in KEYS:
            d = per_q(fired, bm, ce, k) - ce[k]
            pooledC[k].append(d)
            lo, hi = boot_ci(d)
            p(f"   {k:5s}: {100 * d.mean():+5.1f}  95% CI [{lo:+.1f}, {hi:+.1f}]  p={signflip(d):.3f}")
        p()

        # D. probe top-3 vs margin at same cost
        flag = ~agree3
        P = 100
        cost = (3 + (P - 3) * flag.mean()) / P
        kc = int(round(n * cost))
        sm = pick(o_mar, kc, n)
        p(f"D. probe top-3 (keep BM25 if agree) vs margin at the same cost ({100 * cost:.0f}%)")
        for k in KEYS:
            d = per_q(flag, bm, ce, k) - per_q(sm, bm, ce, k)
            pooledD[k].append(d)
            p(f"   {k:5s}: {100 * d.mean():+5.1f}  p={signflip(d):.3f}")
        p()

        # E. tag alarm vs standard label-free confidence signals (lowest confidence reranked first)
        if qpp:
            p("E. tag alarm vs other label-free confidence signals, same budget (Hit@1 / MRR diff, p)")
            for qn, qv in qpp.items():
                o_q = np.argsort(qv, kind="mergesort")
                cells = []
                for b in list(BUDGETS) + ["nat"]:
                    kk = kf if b == "nat" else int(round(n * b / 100))
                    st, sq = pick(o_tag, kk, n), pick(o_q, kk, n)
                    dd = {}
                    for k in KEYS:
                        d = per_q(st, bm, ce, k) - per_q(sq, bm, ce, k)
                        pooledE.setdefault((qn, b, k), []).append(d)
                        dd[k] = d
                    cells.append(f"{('nat' if b == 'nat' else str(b) + '%')}: {100 * dd['Hit@1'].mean():+.1f}/{100 * dd['MRR'].mean():+.1f} (p={signflip(dd['MRR']):.3f})")
                p(f"   vs {qn:20s} " + "  ".join(cells))
            p()

        # G. every selector at the tag alarm's budget vs reranking ALL (does skipping cost accuracy?)
        sels = {"tag alarm": fired, "BM25 margin": pick(o_mar, kf, n)}
        for qn, qv in qpp.items():
            sels[qn] = pick(np.argsort(qv, kind="mergesort"), kf, n)
        p(f"G. each selector reranking {kf} questions ({100 * kf / n:.0f}%) vs reranker on ALL (MRR diff, 95% CI)")
        for sn, sv in sels.items():
            d = per_q(sv, bm, ce, "MRR") - ce["MRR"]
            lo, hi = boot_ci(d)
            p(f"   {sn:20s} {100 * d.mean():+5.1f}  [{lo:+.1f}, {hi:+.1f}]  p={signflip(d):.3f}")
        p()

        # H. full budget curves (no budget taken from the tag alarm) + CSV for the figure
        CURVE = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
        gain = ce["MRR"] - bm["MRR"]
        sel_orders = {"tag alarm": o_tag, "BM25 margin": o_mar}
        for qn, qv in qpp.items():
            sel_orders[qn] = np.argsort(qv, kind="mergesort")
        sel_orders["ORACLE"] = np.argsort(-gain, kind="mergesort")
        allM = {k: ce[k].mean() for k in KEYS}
        p("H. budget curves: MRR at each share reranked; * = no significant loss vs reranking ALL (p >= 0.05 or not lower)")
        p(f"   {'selector':20s} " + " ".join(f"{str(b) + '%':>7s}" for b in CURVE) + "   smallest budget from which there is no significant loss")
        for sn, od in list(sel_orders.items()) + [("random (expected)", None)]:
            cells, oks = [], []
            for b in CURVE:
                kk = int(round(n * b / 100))
                if od is None:
                    r = {k: bm[k].mean() + (kk / n) * (ce[k] - bm[k]).mean() for k in KEYS}
                    pv = float("nan")
                    ok = r["MRR"] >= allM["MRR"] - 1e-12
                else:
                    sv = pick(od, kk, n)
                    r = {k: per_q(sv, bm, ce, k).mean() for k in KEYS}
                    d = per_q(sv, bm, ce, "MRR") - ce["MRR"]
                    pv = signflip(d, n=5000)
                    ok = d.mean() >= -1e-12 or pv >= 0.05
                oks.append(ok)
                curve_rows.append({"dataset": name, "selector": sn, "budget": b, "Hit@1": 100 * r["Hit@1"],
                                   "MRR": 100 * r["MRR"], "dMRR_vs_all": 100 * (r["MRR"] - allM["MRR"]),
                                   "p_vs_all": pv, "bm25_MRR": 100 * bm["MRR"].mean(), "all_MRR": 100 * allM["MRR"]})
                cells.append(f"{100 * r['MRR']:6.1f}{'*' if ok else ' '}")
            first = next((CURVE[i] for i in range(len(CURVE)) if all(oks[i:])), None)
            p(f"   {sn:20s} " + " ".join(cells) + f"   {str(first) + '%' if first else '-'}")
        p(f"   (BM25 only {100 * bm['MRR'].mean():.1f}; rerank ALL {100 * allM['MRR']:.1f})")
        p()

        # I. non-inferiority vs reranking ALL, margin fixed in advance (delta in MRR points)
        def lower_ci(d, nb=2000, seed=1):
            r_ = np.random.default_rng(seed)
            idx = r_.integers(0, len(d), size=(nb, len(d)))
            return float(np.percentile(d[idx].mean(axis=1), 2.5))
        for delta in args.ni_margins:
            p(f"I. NON-INFERIORITY vs reranking ALL, margin delta = {delta} MRR point(s): mean loss [95% CI lower bound]; "
              f"# = non-inferior (lower bound > -delta)")
            p(f"   {'selector':20s} " + " ".join(f"{str(b) + '%':>13s}" for b in CURVE[:-1]) + "   smallest NI budget")
            for sn, od in sel_orders.items():
                cells, oks = [], []
                for b in CURVE[:-1]:
                    sv = pick(od, int(round(n * b / 100)), n)
                    d = per_q(sv, bm, ce, "MRR") - ce["MRR"]
                    lo = 100 * lower_ci(d)
                    ok = lo > -delta
                    oks.append(ok)
                    cells.append(f"{100 * d.mean():+5.1f}[{lo:+5.1f}]{'#' if ok else ' '}")
                oks.append(True)  # 100% = identical to reranking all
                first = next((CURVE[i] for i in range(len(CURVE)) if all(oks[i:])), None)
                p(f"   {sn:20s} " + " ".join(f"{c:>13s}" for c in cells) + f"   {first}%")
            p()

        # J. held-out budget calibration: choose the budget on a small labelled sample, apply it to the rest
        cal_sel = {k: v for k, v in sel_orders.items() if k != "ORACLE"}
        p(f"J. HELD-OUT BUDGET CALIBRATION ({args.cal_reps} random splits; margin delta = {args.ni_margins[0]}): "
          "budget = smallest b with calibration loss >= -delta for all larger b; then applied to the held-out queries")
        p(f"   {'selector':20s} {'n_cal':>6s} {'chosen budget':>16s} {'held-out loss':>15s} {'P(loss < -delta)':>17s} {'saving':>7s}")
        delta = args.ni_margins[0] / 100.0
        sigvals = {"BM25 margin": margin, "tag alarm": None}
        for qn, qv in qpp.items():
            sigvals[qn] = qv
        rng_c = np.random.default_rng(7)
        for sn in cal_sel:
            for ncal in args.cal_sizes:
                if ncal >= n:
                    continue
                bs, losses = [], []
                for _ in range(args.cal_reps):
                    perm = rng_c.permutation(n)
                    C, H = perm[:ncal], perm[ncal:]

                    def order_in(idx):
                        if sn == "tag alarm":
                            return idx[np.lexsort((margin[idx], -alarm[idx]))]
                        return idx[np.argsort(sigvals[sn][idx], kind="mergesort")]
                    oc, oh = order_in(C), order_in(H)
                    chosen = 100
                    ok_all = True
                    for b in reversed(CURVE[:-1]):
                        sel = oc[: int(round(len(C) * b / 100))]
                        m_ = bm["MRR"][C].copy()
                        pos = {q: i for i, q in enumerate(C)}
                        m_[[pos[q] for q in sel]] = ce["MRR"][sel]
                        ok_all = ok_all and (m_.mean() - ce["MRR"][C].mean() >= -delta)
                        if ok_all:
                            chosen = b
                        else:
                            break
                    selh = oh[: int(round(len(H) * chosen / 100))]
                    mh = bm["MRR"][H].copy()
                    posh = {q: i for i, q in enumerate(H)}
                    mh[[posh[q] for q in selh]] = ce["MRR"][selh]
                    bs.append(chosen)
                    losses.append(mh.mean() - ce["MRR"][H].mean())
                bs, losses = np.array(bs), np.array(losses)
                p(f"   {sn:20s} {ncal:6d} {np.mean(bs):7.0f}% (sd {np.std(bs):3.0f}) {100 * losses.mean():+7.2f} MRR "
                  f"{100 * np.mean(losses < -delta):15.0f}% {100 - np.mean(bs):6.0f}%")
        p()

        # F. by query type (selections made on the whole dataset, compared within each type)
        qt = load_qtypes(ds, args.tag)
        types = [t for t in dict.fromkeys(qt) if t]
        if len(types) > 1:
            p("F. tag alarm vs other selectors WITHIN each query type (same total budget; MRR diff, p)")
            for t in types:
                m = qt == t
                cells = []
                for sn, sv in sels.items():
                    if sn == "tag alarm":
                        continue
                    d = (per_q(fired, bm, ce, "MRR") - per_q(sv, bm, ce, "MRR"))[m]
                    cells.append(f"vs {sn}: {100 * d.mean():+.1f} (p={signflip(d):.3f})")
                rr = per_q(fired, bm, ce, "MRR")[m].mean()
                p(f"   {t:14s} n={int(m.sum()):4d}  fires {100 * fired[m].mean():3.0f}%  MRR {100 * rr:5.1f} | " + "  ".join(cells))
            if "named entity" in types and "question" in types and "WIG-like" in sels:
                m = (qt == "named entity") | (qt == "question")
                d = (per_q(fired, bm, ce, "MRR") - per_q(sels["WIG-like"], bm, ce, "MRR"))[m]
                lo, hi = boot_ci(d)
                p(f"   PRIMARY (pre-registered): named entity + question, tag alarm vs WIG-like, MRR "
                  f"{100 * d.mean():+.1f}  [{lo:+.1f}, {hi:+.1f}]  p={signflip(d):.3f}  (n={int(m.sum())})")
            p()

    if len(args.datasets) > 1:
        p(f"## POOLED over {len(args.datasets)} datasets (each dataset uses its own selection)\n")
        p("A. tag alarm vs random (pooled; random draws made within each dataset)")
        for k in KEYS:
            t = np.concatenate(pooled["A_tag"][k]).mean()
            N = sum(len(x) for x in pooled["A_tag"][k])
            parts = []
            for ds_i, ds in enumerate(args.datasets):
                bm, ce, alarm, margin, _, _ = load(ds, args.tag)
                parts.append((ce[k] - bm[k], int((alarm > 0).sum()), bm[k].sum()))
            draws = np.zeros(10000)
            for g, kf, base in parts:
                n = len(g)
                draws += np.array([g[rng.permutation(n)[:kf]].sum() for _ in range(10000)]) + base
            draws /= N
            pv = (np.sum(draws >= t - 1e-12) + 1) / (len(draws) + 1)
            p(f"   {k:5s}: tag {100 * t:5.1f}  random {100 * draws.mean():5.1f}  diff {100 * (t - draws.mean()):+5.1f}  p={pv:.4f}")
        p()
        p("B. tag alarm vs margin, same budget (pooled paired sign-flip)")
        for b in list(BUDGETS) + ["nat"]:
            cells = []
            for k in KEYS:
                d = np.concatenate(pooledB[b][k])
                cells.append(f"{k} {100 * d.mean():+5.1f} (p={signflip(d):.3f})")
            p(f"   {('natural' if b == 'nat' else str(b) + '%'):8s} " + "   ".join(cells))
        p()
        p("C. tag alarm vs rerank ALL (pooled)")
        for k in KEYS:
            d = np.concatenate(pooledC[k])
            lo, hi = boot_ci(d)
            p(f"   {k:5s}: {100 * d.mean():+5.1f}  95% CI [{lo:+.1f}, {hi:+.1f}]  p={signflip(d):.3f}")
        p()
        p("D. probe top-3 vs margin, same cost (pooled)")
        for k in KEYS:
            d = np.concatenate(pooledD[k])
            p(f"   {k:5s}: {100 * d.mean():+5.1f}  p={signflip(d):.3f}")

        if pooledE:
            p()
            p("E. tag alarm vs other label-free signals (pooled; Hit@1 / MRR diff, p on MRR)")
            for qn in dict.fromkeys(key[0] for key in pooledE):
                cells = []
                for b in list(BUDGETS) + ["nat"]:
                    h = np.concatenate(pooledE[(qn, b, "Hit@1")]); m = np.concatenate(pooledE[(qn, b, "MRR")])
                    cells.append(f"{('nat' if b == 'nat' else str(b) + '%')}: {100 * h.mean():+.1f}/{100 * m.mean():+.1f} (p={signflip(m):.3f})")
                p(f"   vs {qn:20s} " + "  ".join(cells))

    if curve_rows:
        import csv
        cfn = os.path.splitext(args.out)[0] + "_curves.csv"
        with open(cfn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(curve_rows[0]))
            w.writeheader()
            w.writerows(curve_rows)
        print(f"saved {cfn}  (for plot_curves.py)")

    with open(args.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
