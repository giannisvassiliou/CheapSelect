#!/usr/bin/env python3
"""
groups46.py: helped/hurt decomposition of the queries WIG skips, per query group (review item 7).
Groups, fixed in advance and observable without labels:
  qtype       DBpedia-Entity query categories (INEX_LD, ListSearch, QALD2, SemSearch_ES), where available
  tag         tag alarm fired (alarm > 0) vs not
  length      query-length terciles (number of distinct query terms, same tokenizer as BM25)
For each group at the policy "WIG re-ranks the lowest b% of the collection" (default b = 50): group size, share of the
group skipped, helped/hurt shares among its skipped queries, and the GROUP LOSS = mean over the group of g_q * 1[skipped]
(what that group loses against re-ranking all; positive = worse), with a 95% bootstrap interval.
Usage: python groups46.py mag_testfull:full prime_testfull:full dbpedia_test:dbp ... --out g46
"""
import argparse
import csv
import json
import os
import re

import numpy as np

from common46 import load

_TOK = re.compile(r"[a-z0-9]+")
_STOP = set("""a an and are as at be by can could do does for from has have how in is it its of on or that the their them
these this those to was were what which who whom whose with""".split())


def qlen(text):
    return len({t for t in _TOK.findall(text.lower()) if t not in _STOP})


def ci(x, nb=2000, seed=3):
    if len(x) == 0:
        return (float("nan"), float("nan"))
    r = np.random.default_rng(seed)
    ii = r.integers(0, len(x), size=(nb, len(x)))
    m = x[ii].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+")
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--out", default="g46")
    a = ap.parse_args()
    rows, md = [], [f"# groups46: WIG re-ranks the lowest {a.budget}% ; group loss = mean_g(gain * skipped), MRR points, 95% bootstrap CI\n"]
    for p in a.pairs:
        D = load(p)
        folder = p.split(":")[0]
        text = {}
        for l in open(os.path.join(folder, "queries.jsonl"), encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                text[str(r["qid"])] = r["query"]
        g, n = D["g"], D["n"]
        o = np.argsort(D["sig"]["WIG-like"], kind="mergesort")
        skip = np.ones(n, bool)
        skip[o[:int(round(n * a.budget / 100))]] = False
        loss = np.where(skip, g, 0.0)
        L = np.array([qlen(text.get(q, "")) for q in D["qid"]])
        cuts = np.quantile(L, [1 / 3, 2 / 3])
        groups = {"all": np.ones(n, bool)}
        if any(D["qtype"]):
            for t in sorted(set(D["qtype"])):
                groups[f"qtype={t}"] = np.array([x == t for x in D["qtype"]])
        groups["tag alarm fired"] = D["alarm"] > 0
        groups["tag alarm silent"] = D["alarm"] <= 0
        groups[f"length short (<= {cuts[0]:.0f})"] = L <= cuts[0]
        groups[f"length medium"] = (L > cuts[0]) & (L <= cuts[1])
        groups[f"length long (> {cuts[1]:.0f})"] = L > cuts[1]
        md.append(f"## {p} (n = {n}; overall loss at {a.budget}% = {100 * loss.mean():+.2f})")
        md.append("   group                         n   skipped  helped/hurt among skipped   group loss [95% CI]")
        for name, m in groups.items():
            if m.sum() == 0:
                continue
            gs = g[m & skip]
            lo, hi = ci(loss[m])
            hh = f"{100 * (gs > 0).mean():4.0f}% / {100 * (gs < 0).mean():3.0f}%" if len(gs) else "   -"
            md.append(f"   {name:28s} {m.sum():5d}  {100 * skip[m].mean():5.0f}%   {hh:>22s}      {100 * loss[m].mean():+6.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]")
            rows.append({"pair": p, "group": name, "n": int(m.sum()), "share_skipped": round(100 * skip[m].mean(), 1),
                         "helped_among_skipped": round(100 * (gs > 0).mean(), 1) if len(gs) else "",
                         "hurt_among_skipped": round(100 * (gs < 0).mean(), 1) if len(gs) else "",
                         "group_loss": round(100 * loss[m].mean(), 3), "ci_lo": round(100 * lo, 3), "ci_hi": round(100 * hi, 3)})
        md.append("")
    open(a.out + ".md", "w").write("\n".join(md) + "\n")
    with open(a.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n".join(md))


if __name__ == "__main__":
    main()
