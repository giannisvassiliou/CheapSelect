"""common46.py: shared loading and policy helpers for the review46 scripts (no GPU).

Per-query files: <folder>/selective_pq_<tag>.jsonl (one line per test query, in pipeline order).
Gain g_q = MRR(re-ranked) - MRR(first stage), in [-1, 1]; one MRR point = 0.01.
Selectors re-rank LOW scores first; a policy with threshold t re-ranks q iff score_q <= t (ties re-ranked: conservative).
"""
import json
import os

import numpy as np

GRID = np.arange(0, 101, 10)          # budgets in %, now including 0 (re-rank none)
QPP = ("top5 vs rest", "NQC-like spread", "WIG-like", "term coverage top1")


def load(arg, metric="MRR"):
    folder, tag = arg.split(":")
    rows = [json.loads(l) for l in open(os.path.join(folder, f"selective_pq_{tag}.jsonl"))]
    if metric == "ndcg":
        bm = np.array([r["ndcg"]["bm"] for r in rows], float)
        ce = np.array([r["ndcg"]["ce"] for r in rows], float)
    else:
        bm = np.array([r["bm"][metric] for r in rows], float)
        ce = np.array([r["ce"][metric] for r in rows], float)
    margin = np.array([r["margin"] for r in rows], float)
    alarm = np.array([r["alarm"] for r in rows], float)
    sig = {"BM25 margin": margin}
    for k in QPP:
        if k in (rows[0].get("qpp") or {}):
            sig[k] = np.array([r["qpp"][k] for r in rows], float)
    # tag alarm as one score (low = re-rank first): -alarm, ties by margin rank
    mr = (np.argsort(np.argsort(margin, kind="mergesort"), kind="mergesort") + 0.5) / len(margin)
    sig["tag alarm"] = -alarm + mr
    return {"name": os.path.basename(folder.rstrip("/")), "tag": tag, "bm": bm, "ce": ce, "g": ce - bm, "sig": sig,
            "qid": [str(r.get("qid_str", i)) for i, r in enumerate(rows)],
            "qtype": [r.get("qtype", "") for r in rows], "alarm": alarm, "rows": rows, "n": len(rows)}


def order_budget_values(score, g, idx=None):
    """per-budget mean loss (positive = worse than re-ranking all) when the lowest-score share b is re-ranked"""
    idx = np.arange(len(g)) if idx is None else idx
    o = idx[np.argsort(score[idx], kind="mergesort")]
    n = len(o)
    out = []
    for b in GRID:
        k = int(round(n * b / 100))
        skip = np.ones(n, bool)
        skip[:k] = False
        out.append(np.where(skip, g[o], 0.0))
    return out  # list of per-query loss vectors (aligned with o)


def smallest_passing(passes):
    """GRID index of the smallest budget such that it and every larger budget pass (passes[-1], b=100, is True)"""
    j = len(passes) - 1
    while j > 0 and passes[j - 1]:
        j -= 1
    return j


def lower_ci_of_mean(d, nb=2000, seed=1):
    r = np.random.default_rng(seed)
    ii = r.integers(0, len(d), size=(nb, len(d)))
    return float(np.percentile(d[ii].mean(axis=1), 2.5))
