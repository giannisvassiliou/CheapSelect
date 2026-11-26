#!/usr/bin/env python3
"""
timing46.py: clean, repeated cost measurements (review items 3 and 11). GPU.

For one collection:
  1. first stage: BM25 scoring of all records, top-100, and the WIG background term (mean over all records) timed separately
  2. re-ranking cost per query for FULL records (100 candidates), for each re-ranker, --reps repetitions after warm-up;
     input length (tokens, after truncation) per pair; and, to explain the earlier pipeline medians, the same with
     TITLE-ONLY records (the shortened texts that the earlier non-selective runs also re-ranked)
  3. router inference cost: bert-base-uncased classifier forward pass on the query text (batch 1 and batch 64)
  4. calibrated fixed policies, measured end to end on untouched queries: WIG threshold certified with UCB95 on 200
     labelled queries (design thresholds from unlabelled queries, as calib46.py), and analytic random (re-rank with
     probability b); reports realized share re-ranked, measured latency (mean, p50, p95) and realized loss (from the
     per-query file of the same re-ranker, matched by query id)
Usage: CUDA_VISIBLE_DEVICES=6 python timing46.py dbpedia_test --columns title,abstract --labels-tag bge43 \
          --rerankers cross-encoder/ms-marco-MiniLM-L-6-v2 BAAI/bge-reranker-v2-m3 --out t46_dbpedia.json
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch

from combo_search44 import BM25, load_jsonl, make_reranker, row_text, tokenize, top_idx
from calib46 import candidates, certify, Z95


def ms(x):
    return 1000.0 * x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--columns", required=True)
    ap.add_argument("--rerankers", nargs="+", default=["BAAI/bge-reranker-v2-m3"])
    ap.add_argument("--batch", type=int, nargs="+", default=[32])
    ap.add_argument("--labels-tag", default="bge43", help="per-query file of the re-ranker used for the policy run")
    ap.add_argument("--policy-reranker", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-policy", type=int, default=300)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cols = a.columns.split(",")
    rng = random.Random(0)
    corpus = load_jsonl(os.path.join(a.data, "corpus.jsonl"))
    rows = [json.loads(l) for l in open(os.path.join(a.data, f"selective_pq_{a.labels_tag}.jsonl"))]
    qtext = {str(r["qid"]): r for r in load_jsonl(os.path.join(a.data, "queries.jsonl"))}
    lab = {str(r["qid_str"]): r for r in rows}
    qids = list(lab)
    rng.shuffle(qids)
    bm = BM25([" ".join(r["fields"][c] for c in cols if r["fields"].get(c)) for r in corpus])
    full = [row_text(r["fields"], cols, False) for r in corpus]
    title = [row_text(r["fields"], [cols[0]], False) for r in corpus]
    res = {"data": a.data, "n_records": len(corpus), "columns": cols, "reps": a.reps, "gpu": torch.cuda.get_device_name(0)}

    # 1. first stage
    t_bm, t_top, t_wig = [], [], []
    tops = {}
    for q in qids[: a.n]:
        qt = tokenize(qtext[q]["query"])
        t0 = time.perf_counter(); s = bm.score(qt); t1 = time.perf_counter()
        top = top_idx(s, 100); t2 = time.perf_counter()
        _ = float(np.mean(s)); t3 = time.perf_counter()
        t_bm.append(t1 - t0); t_top.append(t2 - t1); t_wig.append(t3 - t2)
        tops[q] = top
    res["first_stage_ms"] = {"bm25_median": ms(np.median(t_bm)), "top100_median": ms(np.median(t_top)),
                             "wig_background_mean_median": ms(np.median(t_wig)), "wig_background_mean_p95": ms(np.percentile(t_wig, 95))}
    print(res["first_stage_ms"], flush=True)

    # 2. re-ranking cost, full vs title-only records
    res["rerank"] = {}
    for name, bs in zip(a.rerankers, a.batch * len(a.rerankers)):
        rr, kind = make_reranker(name, device="cuda", fp16=True)
        tok = getattr(getattr(rr, "model", None), "tokenizer", None) or getattr(rr, "tokenizer", None) or getattr(rr, "tok", None)
        q0 = qids[0]
        rr.predict([(qtext[q0]["query"], full[i]) for i in tops[q0]], batch_size=bs, show_progress_bar=False)  # warm-up
        torch.cuda.synchronize()
        out = {}
        for variant, texts in (("full", full), ("title_only", title)):
            per_rep = []
            for rep in range(a.reps):
                times = []
                for q in qids[: a.n]:
                    pairs = [(qtext[q]["query"], texts[i]) for i in tops[q]]
                    t0 = time.perf_counter(); rr.predict(pairs, batch_size=bs, show_progress_bar=False); torch.cuda.synchronize()
                    times.append(time.perf_counter() - t0)
                per_rep.append((ms(np.median(times)), ms(np.mean(times)), ms(np.percentile(times, 95))))
            lens = []
            if tok is not None:
                for q in qids[: min(30, a.n)]:
                    enc = tok([qtext[q]["query"]] * 100, [texts[i] for i in tops[q]], truncation=True, max_length=512)
                    lens += [len(x) for x in enc["input_ids"]]
            per_rep = np.array(per_rep)
            out[variant] = {"median_ms_per_rep": per_rep[:, 0].round(1).tolist(), "mean_ms_per_rep": per_rep[:, 1].round(1).tolist(),
                            "p95_ms_per_rep": per_rep[:, 2].round(1).tolist(), "median_ms": float(per_rep[:, 0].mean()),
                            "median_ms_sd_over_reps": float(per_rep[:, 0].std(ddof=1)) if a.reps > 1 else 0.0,
                            "mean_input_tokens": float(np.mean(lens)) if lens else None, "batch_size": bs}
            print(name, variant, out[variant], flush=True)
        res["rerank"][name] = out
        del rr
        torch.cuda.empty_cache()

    # 3. router inference cost (BERT-base on the query text)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    btok = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3).cuda().eval()
    texts = [qtext[q]["query"] for q in qids[: a.n]]
    with torch.no_grad():
        bert(**{k: v.cuda() for k, v in btok(texts[:2], padding=True, truncation=True, max_length=128, return_tensors="pt").items()})
        t1 = []
        for t in texts:
            e = {k: v.cuda() for k, v in btok([t], truncation=True, max_length=128, return_tensors="pt").items()}
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                bert(**e)
            torch.cuda.synchronize(); t1.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        for s in range(0, len(texts), 64):
            e = {k: v.cuda() for k, v in btok(texts[s:s + 64], padding=True, truncation=True, max_length=128, return_tensors="pt").items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                bert(**e)
        torch.cuda.synchronize()
        tb = (time.perf_counter() - t0) / len(texts)
    res["router_bert_ms"] = {"batch1_median": ms(np.median(t1)), "batch1_p95": ms(np.percentile(t1, 95)), "batch64_per_query": ms(tb)}
    print(res["router_bert_ms"], flush=True)
    del bert
    torch.cuda.empty_cache()

    # 4. calibrated fixed policies (WIG UCB95 and analytic random), end to end on untouched queries
    g = {q: lab[q]["ce"]["MRR"] - lab[q]["bm"]["MRR"] for q in qids}
    wig = {q: lab[q]["qpp"]["WIG-like"] for q in qids}
    # the timing sample above uses no labels, so the policy part draws from all queries; certification, design pool and
    # test are disjoint. Small collections (DBpedia: 400 queries) get a smaller certification sample.
    pq = list(qids)
    random.Random(1).shuffle(pq)
    n_cert = min(200, len(pq) // 2)
    C, rest = pq[:n_cert], pq[n_cert:]
    nd = max(50, int(0.3 * len(rest)))
    Dp, T = rest[:nd], rest[nd: nd + a.n_policy]   # unlabelled design pool, untouched test queries
    assert len(Dp) > 0 and len(T) > 0, f"too few queries for the policy run ({len(pq)})"
    gC = np.array([g[q] for q in C])
    ts, _ = candidates(np.array([wig[q] for q in Dp]))
    t_wig = certify(np.array([wig[q] for q in C]), gC, ts, "UCB95", 0.01)
    U = gC.mean() + Z95 * gC.std(ddof=1) / math.sqrt(len(gC))
    b_rand = 1.0
    for bb in range(90, -1, -10):
        if (1 - bb / 100) * U <= 0.01:
            b_rand = bb / 100
        else:
            break
    rr, _ = make_reranker(a.policy_reranker, device="cuda", fp16=True)
    rr.predict([(qtext[T[0]]["query"], full[0])] * 8, batch_size=32, show_progress_bar=False)
    pol = {}
    prng = np.random.default_rng(11)
    # WIG is computed live (so its cost is in the latency) but the decision uses the stored pipeline value, so that it
    # matches the certified threshold exactly
    for name, decide in ((f"WIG (UCB95, {len(C)} labels)", lambda q, s, top, qt: (live_wig(s, top, qt), wig[q] <= t_wig)[1]),
                         (f"random analytic (b={b_rand:.1f})", lambda q, s, top, qt: prng.random() < b_rand),
                         ("re-rank all", lambda q, s, top, qt: True), ("re-rank none", lambda q, s, top, qt: False)):
        lat, rer, loss = [], [], []
        for q in T:
            t0 = time.perf_counter()
            qt = tokenize(qtext[q]["query"])
            s = bm.score(qt)
            top = top_idx(s, 100)
            r_ = decide(q, s, top, qt)
            if r_:
                rr.predict([(qtext[q]["query"], full[i]) for i in top], batch_size=32, show_progress_bar=False)
            torch.cuda.synchronize()
            lat.append(time.perf_counter() - t0)
            rer.append(r_)
            loss.append(0.0 if r_ else g[q])
        pol[name] = {"share_reranked": float(np.mean(rer)), "latency_mean_ms": ms(np.mean(lat)), "latency_p50_ms": ms(np.median(lat)),
                     "latency_p95_ms": ms(np.percentile(lat, 95)), "realized_loss_mrr_points": 100 * float(np.mean(loss)), "n_queries": len(T)}
        print(name, pol[name], flush=True)
    res["policies"] = pol
    res["policy_setup"] = {"certification_labels": len(C), "design_pool_unlabelled": len(Dp), "wig_threshold": t_wig, "U_g": U}
    json.dump(res, open(a.out, "w"), indent=1)
    print("saved", a.out)


def live_wig(s, top, qt):
    """WIG as in the pipeline: (mean top-5 - mean over all records) / sqrt(number of distinct query terms)"""
    return (float(np.mean(s[top[:5]])) - float(np.mean(s))) / math.sqrt(max(len(set(qt)), 1))


if __name__ == "__main__":
    main()
