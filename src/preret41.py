#!/usr/bin/env python3
"""
preret41.py: pre-retrieval query performance predictors (no retrieval, no GPU, no labels).

For every question it computes, from the question text and corpus word statistics only:
  qlen        number of query terms (after the same tokenizer/stopwords as BM25 in combo_search)
  avgIDF      mean  log(N / df)                                   (Cronen-Townsend et al.; He & Ounis 2004)
  maxIDF      max   log(N / df)
  avgICTF     mean  log(T / cf)                                   (Kwok 1996; He & Ounis 2004)
  avgSCQ      mean  (1 + log cf) * log(1 + N / df)                (Zhao, Scholer & Tsegay 2008)
  maxSCQ      max   of the same
  SCS         simplified clarity: sum_t P(t|q) log(P(t|q) / P(t|C)), P(t|C) = cf / T   (He & Ounis 2004)
N = rows, df = rows containing the term, cf = term count in the collection, T = all term occurrences.
Unseen terms get df = cf = 0.5 (smoothing), so they count as very specific.

The rows are written in exactly the order combo_search uses (queries.jsonl order, keeping questions with
at least one answer in the corpus), so line i matches line i of selective_pq_<tag>.jsonl.

Usage (same --columns as the run; default = all fields):
  python preret41.py prime_testfull --columns "name,details.alias,..."
  python preret41.py dbpedia_test --columns title,abstract
Writes <data>/preret.jsonl
"""
import argparse
import json
import math
import os
import re
import time
from collections import Counter
from multiprocessing import Pool

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = set("""a an and are as at be by can could do does for from has have how in is it its
of on or that the their them these this those to was were what which who whom whose with""".split())


def tokenize(text):
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


_QV, _COLS = None, None


def _init(qv, cols):
    global _QV, _COLS
    _QV, _COLS = qv, cols


def _chunk(lines):
    df, cf, T, N = Counter(), Counter(), 0, 0
    for line in lines:
        f = json.loads(line)["fields"]
        cols = _COLS or list(f)
        text = " ".join(str(f[c]) for c in cols if f.get(c))
        toks = tokenize(text)
        T += len(toks)
        N += 1
        hit = [t for t in toks if t in _QV]
        if hit:
            c = Counter(hit)
            cf.update(c)
            df.update(c.keys())
    return df, cf, T, N


def read_chunks(fn, size=5000):
    buf = []
    with open(fn, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                buf.append(line)
                if len(buf) == size:
                    yield buf
                    buf = []
    if buf:
        yield buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--columns", default="", help="comma-separated, same as the run (default: all fields)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    cols = args.columns.split(",") if args.columns else None

    cfn = os.path.join(args.data, "corpus.jsonl")
    ids = set()
    with open(cfn, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.add(str(json.loads(line)["id"]))
    queries = [json.loads(l) for l in open(os.path.join(args.data, "queries.jsonl"), encoding="utf-8") if l.strip()]
    queries = [q for q in queries if any(str(a) in ids for a in q["answers"])]
    qtoks = [tokenize(q["query"]) for q in queries]
    qv = frozenset(t for ts in qtoks for t in ts)
    print(f"{args.data}: {len(ids)} rows, {len(queries)} questions, {len(qv)} distinct query terms "
          f"({time.time() - t0:.0f}s)", flush=True)

    df, cf, T, N = Counter(), Counter(), 0, 0
    with Pool(args.workers, initializer=_init, initargs=(qv, cols)) as pool:
        for i, (d, c, t, n) in enumerate(pool.imap_unordered(_chunk, read_chunks(cfn))):
            df.update(d)
            cf.update(c)
            T += t
            N += n
            if i % 50 == 0:
                print(f"  {N} rows ({time.time() - t0:.0f}s)", flush=True)
    print(f"corpus done: N={N} rows, T={T} terms ({time.time() - t0:.0f}s)", flush=True)

    out = os.path.join(args.data, "preret.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for i, (q, ts) in enumerate(zip(queries, qtoks)):
            if not ts:
                ts = ["<empty>"]
            dfs = [df.get(t, 0) or 0.5 for t in ts]
            cfs = [cf.get(t, 0) or 0.5 for t in ts]
            idf = [math.log(N / d) for d in dfs]
            ictf = [math.log(T / c) for c in cfs]
            scq = [(1 + math.log(c)) * math.log(1 + N / d) if c >= 1 else math.log(1 + N / d)
                   for c, d in zip(cfs, dfs)]
            qc = Counter(ts)
            scs = sum((k / len(ts)) * math.log((k / len(ts)) / ((cf.get(t, 0) or 0.5) / T)) for t, k in qc.items())
            rec = {"idx": i, "qid_str": str(q.get("qid", "")), "qlen": len(ts) if ts != ["<empty>"] else 0,
                   "avgIDF": sum(idf) / len(idf), "maxIDF": max(idf), "avgICTF": sum(ictf) / len(ictf),
                   "avgSCQ": sum(scq) / len(scq), "maxSCQ": max(scq), "SCS": scs}
            f.write(json.dumps(rec) + "\n")
    print(f"saved {out} ({len(queries)} rows, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
