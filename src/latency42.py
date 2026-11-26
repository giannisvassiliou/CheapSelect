#!/usr/bin/env python3
"""
latency42.py: MEASURED end-to-end cost of selective re-ranking (not the budget x median-time model).

Pipeline per query: BM25 over the whole row -> top-100 -> WIG -> re-rank with the re-ranker only if WIG is below the
threshold for budget b (threshold = b-th percentile of WIG on the measured queries) -> final ranking.
Two serving modes, each at budgets 0 (BM25 only), 30, 50, 70, 100 (re-rank all):
  online : one query at a time; per-query latency p50 / p95 / mean, throughput
  batched: groups of --batch-queries queries; one re-ranker call over all selected (query, record) pairs; throughput
It also prints the analytical model (median BM25 + b * median re-rank time) next to the measurement.

Usage (same --columns as the runs):
  CUDA_VISIBLE_DEVICES=7 python latency42.py mag_testfull --columns "title,abstract,..." \
      --reranker BAAI/bge-reranker-v2-m3 --n 300 --out lat42_mag_bge.json
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch

from combo_search42 import BM25, load_jsonl, make_reranker, row_text, tokenize, top_idx


def wig(s_all, top, nq):
    return (float(np.mean(s_all[top[:5]])) - float(np.mean(s_all))) / np.sqrt(max(nq, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--columns", required=True)
    ap.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--budgets", type=int, nargs="+", default=[0, 30, 50, 70, 100])
    ap.add_argument("--batch-queries", type=int, default=32)
    ap.add_argument("--ce-batch-size", type=int, default=32)
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cols = args.columns.split(",")

    corpus = load_jsonl(os.path.join(args.data, "corpus.jsonl"))
    ids = {str(r["id"]) for r in corpus}
    queries = [q for q in load_jsonl(os.path.join(args.data, "queries.jsonl")) if any(str(a) in ids for a in q["answers"])]
    random.Random(args.seed).shuffle(queries)
    queries = queries[: args.n]
    t = time.time()
    bm = BM25([" ".join(r["fields"][c] for c in cols if r["fields"].get(c)) for r in corpus])
    texts = [row_text(r["fields"], cols, False) for r in corpus]
    print(f"{args.data}: {len(corpus)} rows, {len(queries)} measured queries; BM25 index built in {time.time() - t:.0f}s (not counted)")
    rr, kind = make_reranker(args.reranker, device="cuda", fp16=True)
    print(f"re-ranker {args.reranker} ({kind})")

    def first_stage(q):
        qt = tokenize(q["query"])
        s = bm.score(qt)
        top = top_idx(s, args.depth)
        return top, wig(s, top, len(set(qt)))

    def rerank(pairs):
        return np.asarray(rr.predict(pairs, batch_size=args.ce_batch_size, show_progress_bar=False)) if pairs else np.zeros(0)

    # warm-up and WIG thresholds (label-free: percentiles of WIG over the measured queries)
    rerank([(queries[0]["query"], texts[0])] * 8)
    W = np.array([first_stage(q)[1] for q in queries])
    torch.cuda.synchronize()

    # component medians for the analytical model
    t_bm, t_rr = [], []
    for q in queries[: min(50, len(queries))]:
        a = time.perf_counter(); top, _ = first_stage(q); b = time.perf_counter()
        rerank([(q["query"], texts[i]) for i in top]); torch.cuda.synchronize(); c = time.perf_counter()
        t_bm.append(b - a); t_rr.append(c - b)
    m_bm, m_rr = 1000 * float(np.median(t_bm)), 1000 * float(np.median(t_rr))
    print(f"medians: first stage + WIG {m_bm:.1f} ms, re-rank {args.depth} records {m_rr:.1f} ms\n")

    results = {"data": args.data, "reranker": args.reranker, "n": len(queries), "median_first_stage_ms": m_bm,
               "median_rerank_ms": m_rr, "online": {}, "batched": {}}
    print(f"{'budget':>6s} | {'online p50':>10s} {'p95':>8s} {'mean':>8s} {'q/s':>7s} | {'batched q/s':>11s} | "
          f"{'model mean':>10s} {'measured/model':>14s}")
    for b in args.budgets:
        thr = -np.inf if b == 0 else (np.inf if b == 100 else float(np.percentile(W, b)))
        # online
        lat = []
        t0 = time.perf_counter()
        for q in queries:
            a = time.perf_counter()
            top, w = first_stage(q)
            if w <= thr:
                sc = rerank([(q["query"], texts[i]) for i in top])
                _ = top[np.argsort(-sc, kind="mergesort")]
            torch.cuda.synchronize()
            lat.append(time.perf_counter() - a)
        tot_on = time.perf_counter() - t0
        lat = 1000 * np.array(lat)
        # batched
        t0 = time.perf_counter()
        for s in range(0, len(queries), args.batch_queries):
            grp = queries[s:s + args.batch_queries]
            fs = [first_stage(q) for q in grp]
            pairs, owners = [], []
            for j, (q, (top, w)) in enumerate(zip(grp, fs)):
                if w <= thr:
                    pairs += [(q["query"], texts[i]) for i in top]
                    owners += [j] * len(top)
            sc = rerank(pairs)
            torch.cuda.synchronize()
        tot_b = time.perf_counter() - t0
        share = float(np.mean(W <= thr))
        model = m_bm + share * m_rr
        results["online"][b] = {"share_reranked": share, "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
                                "mean_ms": float(lat.mean()), "qps": len(queries) / tot_on, "model_mean_ms": model}
        results["batched"][b] = {"qps": len(queries) / tot_b}
        print(f"{b:5d}% | {np.percentile(lat, 50):10.1f} {np.percentile(lat, 95):8.1f} {lat.mean():8.1f} {len(queries) / tot_on:7.2f} | "
              f"{len(queries) / tot_b:11.2f} | {model:10.1f} {lat.mean() / model:14.2f}")
    results["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    print(f"\npeak GPU memory {results['peak_gpu_gb']:.1f} GB")
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
