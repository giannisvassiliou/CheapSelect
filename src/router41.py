#!/usr/bin/env python3
"""
router41.py: (1) pre-retrieval predictors as selectors, (2) TRAINED routers, (3) random with its real budget.
Offline: reads <data>/selective_pq_<tag>.jsonl (+ <data>/preret.jsonl from preret41.py, if present). No GPU.

Question it answers: "how much better can you choose WHICH questions to rerank if you have labels?"
  label-free :  each single signal (as in sel_stats40) and each pre-retrieval predictor (fixed direction)
  labels     :  a router trained to predict the per-question gain g = MRR(reranked) - MRR(BM25)
                  - ridge regression (alpha = 1) and gradient-boosted trees (fixed default settings)
                  - features = the six post-retrieval signals [+ the pre-retrieval predictors],
                    each turned into its percentile rank inside the collection (label-free transform)
                  - settings:  5-fold CV (many labels, in-collection, optimistic)
                               n = 50/100/200 labelled questions, applied to the rest (realistic)
                               other collections only (no labels from the target collection)
  ceiling    :  oracle (true gain)
  floor      :  random, whose expected loss is (1 - b) * mean gain  ->  real budget, not 100%
Nothing is tuned on the test data: model settings, feature transform and predictor directions are fixed here.

Selection rule everywhere: rerank the top-b% of questions by predicted need, BM25 for the rest.
Budgets: NI = smallest budget from which the 95% bootstrap lower bound of (selective - all) > -delta at every
larger budget (as sel_stats40 section I); point = same with the mean loss.  saving_vs_random = random_point - point.

Usage (folder[:tag]; tag defaults to --tag):
  python router41.py amazon_testfull mag_testfull prime_testfull dbpedia_test:dbp --tag full --out r41_stark_bge.md
Writes <out>.md, <out>_budgets.csv (same columns as sel_stats40), <out>_summary.csv, <out>_curves.csv
"""
import argparse
import csv
import json
import os

import numpy as np

KEYS = ("Hit@1", "MRR")
CURVE = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
# pre-retrieval predictors: rerank LOW values first (low specificity = hard), except qlen: long first
PRE_DIR = {"qlen": -1, "avgIDF": 1, "maxIDF": 1, "avgICTF": 1, "avgSCQ": 1, "maxSCQ": 1, "SCS": 1}


# ------------------------------------------------------------------ data
def load(arg, default_tag):
    folder, tag = arg.split(":") if ":" in arg else (arg, default_tag)
    rows = [json.loads(l) for l in open(os.path.join(folder, f"selective_pq_{tag}.jsonl"))]
    bm = {k: np.array([r["bm"][k] for r in rows], float) for k in KEYS}
    ce = {k: np.array([r["ce"][k] for r in rows], float) for k in KEYS}
    sig = {"BM25 margin": np.array([r["margin"] for r in rows], float),
           "tag alarm": np.array([r["alarm"] for r in rows], float)}
    for name in (rows[0].get("qpp") or {}):
        sig[name] = np.array([r["qpp"][name] for r in rows], float)
    pre = {}
    pfn = os.path.join(folder, "preret.jsonl")
    if os.path.exists(pfn):
        pr = [json.loads(l) for l in open(pfn)]
        ok = len(pr) == len(rows)
        if ok and "qid_str" in rows[0]:
            ok = all(str(a.get("qid_str", "")) == str(b["qid_str"]) for a, b in zip(rows, pr))
        if ok:
            pre = {k: np.array([p[k] for p in pr], float) for k in PRE_DIR}
        else:
            print(f"WARNING {folder}: preret.jsonl does not line up with selective_pq_{tag}.jsonl - ignored")
    return {"name": os.path.basename(folder.rstrip("/")), "tag": tag, "bm": bm, "ce": ce, "sig": sig,
            "pre": pre, "n": len(rows), "gain": ce["MRR"] - bm["MRR"]}


def pct(x):
    """percentile rank in [0, 1], ties averaged"""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    r = (np.bincount(inv, weights=r) / cnt)[inv]
    return r / max(1, len(x) - 1)


def spearman(a, b):
    return float(np.corrcoef(pct(a), pct(b))[0, 1])


def features(D, use_pre):
    names = list(D["sig"]) + (list(D["pre"]) if use_pre else [])
    cols = [pct(D["sig"][k]) for k in D["sig"]] + ([pct(D["pre"][k]) for k in D["pre"]] if use_pre else [])
    return np.column_stack(cols), names


# ------------------------------------------------------------------ models
def ridge_fit(X, y, alpha=1.0):
    Xb = np.column_stack([np.ones(len(X)), X])
    I = np.eye(Xb.shape[1])
    I[0, 0] = 0.0
    w = np.linalg.solve(Xb.T @ Xb + alpha * I, Xb.T @ y)
    return lambda Z: np.column_stack([np.ones(len(Z)), Z]) @ w, w


def gbm_fit(X, y):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                      min_samples_leaf=20, random_state=0).fit(X, y)
    return lambda Z: m.predict(Z), None


MODELS = {"ridge": ridge_fit, "gbm": gbm_fit}


def cv_scores(X, y, fit, folds=5, seed=0):
    n = len(y)
    perm = np.random.default_rng(seed).permutation(n)
    s = np.zeros(n)
    for f in range(folds):
        te = perm[f::folds]
        tr = np.setdiff1d(perm, te)
        pred, _ = fit(X[tr], y[tr])
        s[te] = pred(X[te])
    return s


# ------------------------------------------------------------------ evaluation
def order_of(name, D):
    if name == "tag alarm":
        return np.lexsort((D["sig"]["BM25 margin"], -D["sig"]["tag alarm"]))
    return np.argsort(D["sig"][name], kind="mergesort")


def sel_values(order, D, idx=None, b=50):
    """per-question MRR/Hit@1 when the first b% of `order` (restricted to idx) is reranked"""
    idx = np.arange(D["n"]) if idx is None else idx
    o = order if idx is None else order
    k = int(round(len(idx) * b / 100))
    chosen = np.zeros(D["n"], bool)
    chosen[o[:k]] = True
    return {m: np.where(chosen, D["ce"][m], D["bm"][m])[idx] for m in KEYS}


def lower_ci(d, nb=2000, seed=1):
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(d), size=(nb, len(d)))
    return float(np.percentile(d[idx].mean(axis=1), 2.5))


def signflip(d, n=5000, seed=0):
    d = np.asarray(d, float)
    if not np.any(d):
        return 1.0
    rng = np.random.default_rng(seed)
    null = np.abs((rng.choice([-1.0, 1.0], size=(n, len(d))) * d).mean(axis=1))
    return float((np.sum(null >= abs(d.mean()) - 1e-12) + 1) / (n + 1))


def budgets(order, D, deltas, idx=None):
    idx = np.arange(D["n"]) if idx is None else idx
    allm = D["ce"]["MRR"][idx]
    losses, los = [], []
    for b in CURVE[:-1]:
        d = sel_values(order, D, idx, b)["MRR"] - allm
        losses.append(float(d.mean()))
        los.append(lower_ci(d))
    losses.append(0.0)
    los.append(0.0)
    out = {}
    for de in deltas:
        de_ = de / 100
        ni = next(CURVE[i] for i in range(len(CURVE)) if all(x > -de_ for x in los[i:]))
        pt = next(CURVE[i] for i in range(len(CURVE)) if all(x >= -de_ - 1e-12 for x in losses[i:]))
        out[de] = (ni, pt)
    return out, losses


def random_point(D, delta, n_sub=None):
    g = D["gain"].mean()
    n = n_sub or D["n"]
    return next(b for b in CURVE if all((1 - int(round(n * bb / 100)) / n) * g <= delta / 100 + 1e-12
                                        for bb in CURVE if bb >= b))


def restrict(order, idx):
    keep = np.zeros(len(order), bool)
    keep[idx] = True
    return order[keep[order]]


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--out", default="router41.md")
    ap.add_argument("--deltas", type=float, nargs="+", default=[1.0, 0.5])
    ap.add_argument("--nlab", type=int, nargs="+", default=[50, 100, 200])
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--no-gbm", action="store_true")
    args = ap.parse_args()
    models = {k: v for k, v in MODELS.items() if not (k == "gbm" and args.no_gbm)}
    if "gbm" in models:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            print("sklearn missing - gradient-boosted router skipped")
            models.pop("gbm")

    out, budget_rows, summary_rows, curve_rows = [], [], [], []

    def p(s=""):
        print(s)
        out.append(s)

    data = [load(a, args.tag) for a in args.datasets]
    p(f"# router41: label-free signals vs pre-retrieval vs trained routers  (deltas {args.deltas} MRR points)\n")

    for D in data:
        n, g = D["n"], D["gain"]
        allM = {k: D["ce"][k].mean() for k in KEYS}
        p(f"## {D['name']} (tag {D['tag']}; n={n}; mean gain {100 * g.mean():+.1f} MRR; "
          f"helped {100 * (g > 0).mean():.0f}% hurt {100 * (g < 0).mean():.0f}%)\n")

        # ---- pre-retrieval correlations (descriptive)
        if D["pre"]:
            p("Pre-retrieval predictors, Spearman rho with BM25 quality and with gain "
              "(oriented: higher = 'easy, skip the reranker')")
            for k, v in D["pre"].items():
                vv = PRE_DIR[k] * v
                p(f"   {k:8s} rho(quality) {spearman(vv, D['bm']['MRR']):+.2f}   rho(gain) {spearman(vv, g):+.2f}")
            p()

        # ---- all selectors on the full collection
        orders = {}
        for s in D["sig"]:
            orders[s] = order_of(s, D)
        for k, v in D["pre"].items():
            orders[f"pre:{k}"] = np.argsort(PRE_DIR[k] * v, kind="mergesort")
        coefs = {}
        for use_pre in ([False, True] if D["pre"] else [False]):
            X, fn = features(D, use_pre)
            fs = "S+P" if use_pre else "S"
            for mn, fit in models.items():
                orders[f"router {mn} {fs} (5-fold CV)"] = np.argsort(-cv_scores(X, g, fit), kind="mergesort")
            coefs[fs] = (fn, ridge_fit(X, g)[1])
        if len(data) > 1:
            others = [E for E in data if E is not D]
            for use_pre in ([False, True] if (D["pre"] and all(E["pre"] for E in others)) else [False]):
                fs = "S+P" if use_pre else "S"
                common = [k for k in D["sig"] if all(k in E["sig"] for E in others)]
                pre_k = list(D["pre"]) if use_pre else []

                def fx(E):
                    return np.column_stack([pct(E["sig"][k]) for k in common] + [pct(E["pre"][k]) for k in pre_k])
                Xtr = np.vstack([fx(E) for E in others])
                ytr = np.concatenate([E["gain"] for E in others])
                for mn, fit in models.items():
                    pred, _ = fit(Xtr, ytr)
                    orders[f"router {mn} {fs} (other collections)"] = np.argsort(-pred(fx(D)), kind="mergesort")
        orders["ORACLE"] = np.argsort(-g, kind="mergesort")

        wig = next((s for s in D["sig"] if s.lower().startswith("wig")), None)
        rp = {de: random_point(D, de) for de in args.deltas}
        p(f"Budgets (share of questions reranked). random expected point budget: "
          + ", ".join(f"delta {de}: {rp[de]}%" for de in args.deltas))
        hdr = "   " + f"{'selector':38s}" + "".join(f"  NI/pt d={de:<4}" for de in args.deltas) + \
              "  save>rand(d=1st)  sel-all@50%  vs WIG@50% (p)"
        p(hdr)
        for sn, od in orders.items():
            bd, losses = budgets(od, D, args.deltas)
            v50 = sel_values(od, D, None, 50)["MRR"]
            l50 = 100 * (v50 - D["ce"]["MRR"]).mean()
            if wig and sn != wig:
                dw = v50 - sel_values(orders[wig], D, None, 50)["MRR"]
                vs = f"{100 * dw.mean():+5.1f} ({signflip(dw):.3f})"
            else:
                vs = "      -"
            d0 = args.deltas[0]
            p("   " + f"{sn:38s}" + "".join(f"  {bd[de][0]:>4}/{bd[de][1]:<4}    " for de in args.deltas)
              + f"  {rp[d0] - bd[d0][1]:+6d}           {l50:+6.1f}    {vs}")
            for de in args.deltas:
                if sn in D["sig"]:
                    pass  # base selectors are already in sel_stats40's budgets csv
                else:
                    budget_rows.append({"dataset": D["name"], "selector": sn, "delta": de, "ni_budget": bd[de][0],
                                        "point_budget": bd[de][1], "n": n})
                summary_rows.append({"dataset": D["name"], "tag": D["tag"], "selector": sn, "delta": de,
                                     "ni_budget": bd[de][0], "point_budget": bd[de][1],
                                     "random_point_budget": rp[de], "saving_vs_random": rp[de] - bd[de][1],
                                     "loss_at_50": round(l50, 2), "mean_gain": round(100 * g.mean(), 2), "n": n})
            if sn not in D["sig"] and sn != "ORACLE":
                for b in CURVE:
                    r = sel_values(od, D, None, b)
                    curve_rows.append({"dataset": D["name"], "selector": sn, "budget": b,
                                       "Hit@1": 100 * r["Hit@1"].mean(), "MRR": 100 * r["MRR"].mean(),
                                       "dMRR_vs_all": 100 * (r["MRR"].mean() - allM["MRR"]), "p_vs_all": float("nan"),
                                       "bm25_MRR": 100 * D["bm"]["MRR"].mean(), "all_MRR": 100 * allM["MRR"]})
        for de in args.deltas:
            budget_rows.append({"dataset": D["name"], "selector": "random (expected)", "delta": de,
                                "ni_budget": "", "point_budget": rp[de], "n": n})
        p()
        for fs, (fn, w) in coefs.items():
            p(f"Ridge weights, features {fs}, fit on all questions (descriptive; + = more gain expected): "
              + ", ".join(f"{a} {b:+.3f}" for a, b in zip(fn, w[1:])))
        p()

        # ---- few labels: train on n labelled questions, test on the rest, compare to WIG on the same rest
        if wig:
            p(f"Few labels: router trained on n labelled questions, evaluated on the other questions "
              f"({args.reps} random splits). Loss vs reranking all on held-out (MRR points), at budgets 30/50/70%; "
              f"P(router > {wig} at 50%)")
            X, _ = features(D, bool(D["pre"]))
            fs = "S+P" if D["pre"] else "S"
            for nl in args.nlab:
                if nl >= n - 50:
                    continue
                res = {mn: [] for mn in models}
                resw, beat = [], {mn: [] for mn in models}
                rng = np.random.default_rng(11)
                for _ in range(args.reps):
                    perm = rng.permutation(n)
                    L, H = perm[:nl], perm[nl:]
                    ow = restrict(orders[wig], H)
                    rw = [float((sel_values(ow, D, H, b)["MRR"] - D["ce"]["MRR"][H]).mean()) for b in (30, 50, 70)]
                    resw.append(rw)
                    for mn, fit in models.items():
                        pred, _ = fit(X[L], g[L])
                        sc = np.full(n, -np.inf)
                        sc[H] = pred(X[H])
                        orr = restrict(np.argsort(-sc, kind="mergesort"), H)
                        rr = [float((sel_values(orr, D, H, b)["MRR"] - D["ce"]["MRR"][H]).mean()) for b in (30, 50, 70)]
                        res[mn].append(rr)
                        beat[mn].append(rr[1] > rw[1])
                w_ = 100 * np.mean(resw, axis=0)
                p(f"   n={nl:4d}  {wig:12s}            " + "  ".join(f"{x:+5.1f}" for x in w_))
                for mn in models:
                    m_ = 100 * np.mean(res[mn], axis=0)
                    p(f"   n={nl:4d}  router {mn:5s} {fs:4s}      " + "  ".join(f"{x:+5.1f}" for x in m_)
                      + f"   P(router > {wig} @50%) = {100 * np.mean(beat[mn]):.0f}%")
            p()

    for fn_, rows in (("_budgets.csv", budget_rows), ("_summary.csv", summary_rows), ("_curves.csv", curve_rows)):
        if rows:
            path = os.path.splitext(args.out)[0] + fn_
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            print(f"saved {path}")
    with open(args.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
