#!/usr/bin/env python3
"""
genc45.py: re-implementation of the supervised router of Genc et al. (ICTIR'26, "Adaptive Re-Ranking") on our data,
and comparison with label-free WIG selection at equal cost.

Their recipe (paper + released labelling code, github.com/emirkaan5/adaptive-reranker):
  classes   0 = BM25 only, 1 = BM25 + MiniLM-L6, 2 = BM25 + bge-reranker-v2-m3
  label     argmax_m  eff_m - 0.05 * lat_m / max_m' lat_m'     eff = (nDCG@10 + MRR@10) / 2,
            lat_m = average per-query latency of model m on the collection (re-ranker time only; BM25 time for class 0)
  router    bert-base-uncased sequence classifier on the query text; training set down-sampled to the minority class;
            AdamW lr 3e-5, 20 epochs, checkpoint with the best validation accuracy
Our data: per-query nDCG@10 / MRR of BM25, MiniLM and bge from the v43 runs (tags mini43, bge43), same queries.
Two training regimes (they trained on ~300K BEIR training queries, which we do not have):
  cv5  : 5-fold cross-validation within each collection (10% of each training fold for validation)
  loco : train on all other collections, test on the held-out one (their "out-of-domain" setting)
Evaluation per collection:
  (a) their view: nDCG@10 and mean latency of the routed pipeline vs BM25 / MiniLM / bge on all queries, oracle, uniform random
  (b) iso-cost label-free baselines with the router's own class shares: WIG-3way (lowest WIG -> bge, next -> MiniLM, rest BM25)
      and random-3way; and WIG-2way (bge or BM25) at the router's mean latency
  (c) our view: router as a selector (order by P(bge) - P(BM25)); point budget vs bge-on-all at delta = 1 nDCG point, next to WIG's

Usage (GPU, ~1-2 h):  CUDA_VISIBLE_DEVICES=6 python genc45.py --lat lat45.json --out g45
   collections are fixed below; --lat: {"mag_testfull": {"bm25": ms, "mini": ms, "bge": ms}, ...}
"""
import argparse
import csv
import json
import os
import random

import numpy as np

COLL = ["amazon_testfull", "mag_testfull", "prime_testfull", "dbpedia_test",
        "beir_scifact", "beir_nfcorpus", "beir_fiqa", "beir_arguana", "beir_scidocs"]
LAMBDA = 0.05


def load(folder):
    q = {}
    for l in open(os.path.join(folder, "queries.jsonl"), encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            q[str(r["qid"])] = r["query"]
    M = [json.loads(l) for l in open(os.path.join(folder, "selective_pq_mini43.jsonl"))]
    B = [json.loads(l) for l in open(os.path.join(folder, "selective_pq_bge43.jsonl"))]
    assert len(M) == len(B) and all(a["qid_str"] == b["qid_str"] for a, b in zip(M, B)), folder

    def mrr10(x):
        return x if x >= 0.1 - 1e-12 else 0.0
    nd = np.array([[b["ndcg"]["bm"], m["ndcg"]["ce"], b["ndcg"]["ce"]] for m, b in zip(M, B)])
    mr = np.array([[mrr10(b["bm"]["MRR"]), mrr10(m["ce"]["MRR"]), mrr10(b["ce"]["MRR"])] for m, b in zip(M, B)])
    text = [q[str(b["qid_str"])] for b in B]
    wig = np.array([b["qpp"]["WIG-like"] for b in B])
    return {"text": text, "ndcg": nd, "mrr": mr, "eff": (nd + mr) / 2, "wig": wig}


def labels(eff, lat):
    l = np.array([lat["bm25"], lat["mini"], lat["bge"]], float)
    u = eff - LAMBDA * l / l.max()
    return np.argmax(u, axis=1)  # ties -> lowest index (cheapest), as max() over the ordered dict in their code


def train_router(texts, y, seed, args):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    nval = max(1, int(0.1 * len(idx)))
    va, tr = idx[:nval], idx[nval:]
    # down-sample training classes to the minority class (classes that occur at all)
    present = [c for c in range(3) if (y[tr] == c).sum() > 0]
    m = min((y[tr] == c).sum() for c in present)
    tr = np.concatenate([rng.choice(tr[y[tr] == c], m, replace=False) for c in present])
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)

    def enc(ii):
        e = tok([texts[i] for i in ii], padding="max_length", truncation=True, max_length=args.max_len, return_tensors="pt")
        return {k: v.cuda() for k, v in e.items()}
    best, best_state = -1, None
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(tr)
        for s in range(0, len(order), args.bs):
            b = order[s:s + args.bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**enc(b), labels=torch.tensor(y[b]).cuda())
            out.loss.backward()
            opt.step()
            opt.zero_grad()
        acc = (predict(model, enc, va, args) .argmax(1) == y[va]).mean()
        if acc > best:
            best = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, enc, best


def predict(model, enc, ii, args):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(ii), 256):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(torch.softmax(model(**enc(ii[s:s + 256])).logits.float(), dim=1).cpu().numpy())
    return np.concatenate(out)


def evaluate(D, cls, prob, lat, delta):
    n = len(cls)
    L = np.array([lat["bm25"], lat["bm25"] + lat["mini"], lat["bm25"] + lat["bge"]])
    nd = D["ndcg"]

    def route(c):
        return 100 * nd[np.arange(n), c].mean(), float(L[c].mean())
    res = {"BM25": route(np.zeros(n, int)), "MiniLM all": route(np.ones(n, int)), "bge all": route(np.full(n, 2)),
           "oracle (their utility)": route(labels(D["eff"], lat)), "router": route(cls)}
    rng = np.random.default_rng(0)
    res["random uniform 1/3"] = route(rng.integers(0, 3, n))
    shares = np.bincount(cls, minlength=3) / n
    o = np.argsort(D["wig"], kind="mergesort")       # low WIG = uncertain first
    k2, k1 = int(round(shares[2] * n)), int(round(shares[1] * n))
    c3 = np.zeros(n, int)
    c3[o[:k2]] = 2
    c3[o[k2:k2 + k1]] = 1
    res["WIG 3-way (router's shares)"] = route(c3)
    cr = np.zeros(n, int)
    p = rng.permutation(n)
    cr[p[:k2]] = 2
    cr[p[k2:k2 + k1]] = 1
    res["random 3-way (router's shares)"] = route(cr)
    b2 = np.clip((res["router"][1] - lat["bm25"]) / lat["bge"], 0, 1)
    c2 = np.zeros(n, int)
    c2[o[:int(round(b2 * n))]] = 2
    res[f"WIG 2-way (same latency, {100 * b2:.0f}% to bge)"] = route(c2)
    # our view: point budget vs bge-on-all (delta in nDCG points)

    def point_budget(order):
        means = []
        for b in range(10, 101, 10):
            sel = np.zeros(n, bool)
            sel[order[:int(round(n * b / 100))]] = True
            means.append(np.where(sel, nd[:, 2], nd[:, 0]).mean() - nd[:, 2].mean())
        return next(10 * (j + 1) for j in range(10) if all(m >= -delta / 100 - 1e-12 for m in means[j:]))
    pb = {"router (P(bge)-P(BM25))": point_budget(np.argsort(-(prob[:, 2] - prob[:, 0]), kind="mergesort")),
          "WIG": point_budget(o)}
    return res, shares, pb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", required=True)
    ap.add_argument("--out", default="g45")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--modes", nargs="+", default=["cv5", "loco"])
    args = ap.parse_args()
    random.seed(0)
    lat = json.load(open(args.lat))
    data = {c: load(c) for c in COLL if os.path.exists(os.path.join(c, "selective_pq_bge43.jsonl"))}
    for c, D in data.items():
        D["y"] = labels(D["eff"], lat[c])
    out, rows = [], []

    def p(s=""):
        print(s, flush=True)
        out.append(s)
    p(f"# genc45: Genc et al.'s router re-implemented on our data (lambda = {LAMBDA}; nDCG@10 in %, latency in ms)\n")
    for c, D in data.items():
        p(f"{c}: n={len(D['y'])}, label shares BM25/MiniLM/bge = " + "/".join(f"{100 * x:.0f}%" for x in np.bincount(D['y'], minlength=3) / len(D['y'])))
    p()
    for mode in args.modes:
        p(f"## mode {mode}")
        for c, D in data.items():
            n = len(D["y"])
            prob = np.zeros((n, 3))
            accs = []
            if mode == "cv5":
                perm = np.random.default_rng(1).permutation(n)
                for f in range(5):
                    te = perm[f::5]
                    tr = np.setdiff1d(perm, te)
                    model, enc_, acc = train_router([D["text"][i] for i in tr], D["y"][tr], f, args)
                    from transformers import AutoTokenizer  # same tokenizer, encode test texts
                    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
                    import torch

                    def enc_t(ii, T=[D["text"][i] for i in te]):
                        e = tok([T[i] for i in ii], padding="max_length", truncation=True, max_length=args.max_len, return_tensors="pt")
                        return {k: v.cuda() for k, v in e.items()}
                    prob[te] = predict(model, enc_t, np.arange(len(te)), args)
                    accs.append(acc)
                    del model
                    torch.cuda.empty_cache()
            else:
                others = [o for o in data if o != c]
                T = [t for o in others for t in data[o]["text"]]
                Y = np.concatenate([data[o]["y"] for o in others])
                model, _, acc = train_router(T, Y, 0, args)
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained("bert-base-uncased")
                import torch

                def enc_t(ii, T=D["text"]):
                    e = tok([T[i] for i in ii], padding="max_length", truncation=True, max_length=args.max_len, return_tensors="pt")
                    return {k: v.cuda() for k, v in e.items()}
                prob = predict(model, enc_t, np.arange(n), args)
                accs.append(acc)
                del model
                torch.cuda.empty_cache()
            cls = prob.argmax(1)
            res, shares, pb = evaluate(D, cls, prob, lat[c], args.delta)
            test_acc = float((cls == D["y"]).mean())
            p(f"### {c}  (router test accuracy {100 * test_acc:.0f}%, val {100 * np.mean(accs):.0f}%; routed shares BM25/MiniLM/bge = "
              + "/".join(f"{100 * x:.0f}%" for x in shares) + ")")
            allb = res["bge all"]
            for k, (v, l) in res.items():
                p(f"   {k:40s} nDCG {v:6.2f} ({100 * (v - allb[0]) / allb[0]:+6.1f}% vs bge)   latency {l:8.1f} ({100 * (l - allb[1]) / allb[1]:+5.0f}%)")
                rows.append({"mode": mode, "dataset": c, "policy": k, "ndcg": round(v, 3), "latency_ms": round(l, 2),
                             "ndcg_rel_bge": round(100 * (v - allb[0]) / allb[0], 2), "lat_rel_bge": round(100 * (l - allb[1]) / allb[1], 2),
                             "router_acc": round(test_acc, 3)})
            p(f"   point budget (delta={args.delta} nDCG pt, bge vs BM25): router {pb['router (P(bge)-P(BM25))']}%   WIG {pb['WIG']}%")
            rows.append({"mode": mode, "dataset": c, "policy": "point budget router/WIG", "ndcg": pb["router (P(bge)-P(BM25))"],
                         "latency_ms": pb["WIG"], "ndcg_rel_bge": "", "lat_rel_bge": "", "router_acc": round(test_acc, 3)})
        p()
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(out) + "\n")
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {args.out}.md / .csv")


if __name__ == "__main__":
    main()
