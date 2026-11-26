#!/usr/bin/env python3
"""
cascade43.py: a cheap re-ranker (MiniLM) as the proxy for an expensive one (bge or monoT5) -- the cascade baseline.

Joins two runs on the same queries and BM25 lists:  --cheap <tag of the MiniLM run>  --exp <tag of the expensive run>.
Policies, each at budgets 10..100% (share of queries sent to the expensive re-ranker):
  BM25 | WIG          : BM25 for skipped queries, selection by WIG                       (the paper's setting)
  MiniLM | WIG        : MiniLM ranking for skipped queries, selection by WIG
  BM25 | MiniLM-dis   : BM25 for skipped, selection by MiniLM disagreement with BM25 (label-free)
  MiniLM | MiniLM-dis : MiniLM for skipped, selection by MiniLM disagreement
  random              : BM25 for skipped, random selection (expected)
MiniLM disagreement = MiniLM's top-1 differs from BM25's top-1 (v43 'cef.agree_top1'), then low MiniLM margin, then
low rank correlation with BM25 on the top-10; with pre-v43 files only the top-3 probe ('probe3_agree') is available.
Reports MRR, the point-estimate budget (mean loss vs expensive-on-all <= delta at every larger budget) and the cost
model  cheap_ms (if MiniLM is run on all queries) + b * exp_ms  (pass --cheap-ms/--exp-ms from the TIMING blocks).

Usage: python cascade43.py mag_testfull prime_testfull amazon_testfull dbpedia_test:dbp_full=full --cheap mini --exp full \
          --cheap-ms 98 80 146 47 --exp-ms 257 211 316 87 --out k43_stark_bge.md
  (a dataset may carry its own expensive tag: dbpedia_test:dbp, i.e. folder:exp_tag)
"""
import argparse
import json
import os

import numpy as np

B = np.arange(10, 101, 10)


def rows(folder, tag):
    return [json.loads(l) for l in open(os.path.join(folder, f"selective_pq_{tag}.jsonl"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--cheap", default="mini")
    ap.add_argument("--exp", default="full")
    ap.add_argument("--cheap-ms", type=float, nargs="*", default=[])
    ap.add_argument("--exp-ms", type=float, nargs="*", default=[])
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--out", default="cascade43.md")
    args = ap.parse_args()
    d = args.delta / 100
    out = []

    def p(s=""):
        print(s)
        out.append(s)

    p(f"# cascade43: cheap re-ranker ({args.cheap}) as proxy / fallback for an expensive one; delta = {args.delta}\n")
    for i, a in enumerate(args.datasets):
        folder, exp_tag = (a.split(":") + [args.exp])[:2]
        C, E = rows(folder, args.cheap), rows(folder, exp_tag)
        if len(C) != len(E) or ("qid_str" in C[0] and "qid_str" in E[0] and
                                any(c["qid_str"] != e["qid_str"] for c, e in zip(C, E))):
            p(f"## {folder}: runs do not line up - skipped\n")
            continue
        n = len(C)
        bm = np.array([r["bm"]["MRR"] for r in E])
        ex = np.array([r["ce"]["MRR"] for r in E])
        ch = np.array([r["ce"]["MRR"] for r in C])
        wig = np.array([r["qpp"]["WIG-like"] for r in E])
        if "cef" in C[0]:
            agree = np.array([r["cef"]["agree_top1"] for r in C], float)
            marg = np.array([r["cef"]["margin"] for r in C], float)
            rho = np.array([r["cef"]["rho10"] for r in C], float)
            dis_order = np.lexsort((rho, marg, agree))   # disagree first, then small margin, then low rho
            kind = "top-1 + margin + rho10 (v43)"
        else:
            agree = np.array([r["probe3_agree"] for r in C], float)
            dis_order = np.lexsort((wig, agree))         # disagree on the top-3 probe first, ties by WIG
            kind = "top-3 probe (pre-v43)"
        wig_order = np.argsort(wig, kind="mergesort")
        allE = ex.mean()
        cm = args.cheap_ms[i] if i < len(args.cheap_ms) else float("nan")
        em = args.exp_ms[i] if i < len(args.exp_ms) else float("nan")
        p(f"## {os.path.basename(folder)} (n={n}; BM25 {100 * bm.mean():.1f}, {args.cheap} all {100 * ch.mean():.1f}, "
          f"{exp_tag} all {100 * allE:.1f}; disagreement = {kind}, rate {100 * (1 - agree.mean()):.0f}%)")
        pol = {"BM25 | WIG": (bm, wig_order), "MiniLM | WIG": (ch, wig_order),
               "BM25 | MiniLM-dis": (bm, dis_order), "MiniLM | MiniLM-dis": (ch, dis_order)}
        p(f"   {'policy':22s} " + " ".join(f"{b:>5d}%" for b in B) + "   point budget   cost at that budget (ms)")
        for name, (base, order) in pol.items():
            vals, means = [], []
            for b in B:
                k = int(round(n * b / 100))
                sel = np.zeros(n, bool)
                sel[order[:k]] = True
                v = np.where(sel, ex, base)
                vals.append(100 * v.mean())
                means.append(v.mean() - allE)
            pt = next(B[j] for j in range(len(B)) if all(m >= -d - 1e-12 for m in means[j:]))
            uses_cheap = name.startswith("MiniLM") or "MiniLM-dis" in name
            cost = (cm if uses_cheap else 0.0) + pt / 100 * em
            p(f"   {name:22s} " + " ".join(f"{v:6.1f}" for v in vals) + f"   {pt:5d}%        {cost:8.0f}  (all: {em:.0f})")
        gbar = (ex - bm).mean()
        rp = next(b for b in B if all((1 - int(round(n * bb / 100)) / n) * gbar <= d + 1e-12 for bb in B if bb >= b))
        p(f"   {'random (expected)':22s} " + " ".join(f"{100 * (bm.mean() + b / 100 * gbar):6.1f}" for b in B) + f"   {rp:5d}%")
        p()
    with open(args.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
