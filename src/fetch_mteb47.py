#!/usr/bin/env python3
"""
fetch_mteb47.py: download a BEIR collection from its MTEB copy on Hugging Face (when the BEIR zip server is unreachable)
and write the plain BEIR layout that prepare_beir.py --beir-dir reads:
   OUT/corpus.jsonl  {"_id", "title", "text"}     OUT/queries.jsonl {"_id", "text"}     OUT/qrels/test.tsv
Handles jsonl, jsonl.gz, tsv and parquet files, whatever the repo uses.
Usage: python fetch_mteb47.py mteb/cqadupstack-english cqadupstack/english
"""
import csv
import gzip
import io
import json
import os
import sys

from huggingface_hub import list_repo_files, snapshot_download


def rows(path):
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        for b in pq.ParquetFile(path).iter_batches(batch_size=50000):
            yield from b.to_pylist()
    elif path.endswith((".tsv", ".tsv.gz")):
        fh = io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")
        yield from csv.DictReader(fh, delimiter="\t")
    else:
        fh = io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")
        for l in fh:
            if l.strip():
                yield json.loads(l)


def main():
    repo, out = sys.argv[1], sys.argv[2]
    files = list_repo_files(repo, repo_type="dataset")
    print("files in", repo, ":", files)
    data = lambda f: f.endswith((".parquet", ".jsonl", ".jsonl.gz", ".tsv", ".tsv.gz"))  # noqa: E731
    corpus = [f for f in files if "corpus" in f.lower() and data(f)]
    queries = [f for f in files if "quer" in f.lower() and data(f)]
    qrels = [f for f in files if data(f) and "test" in f.lower() and f not in corpus and f not in queries]
    # several repos ship the same judgements in two formats (qrels/test.jsonl and qrels/test.tsv): read only one
    for ext in (".tsv", ".jsonl", ".parquet", ".tsv.gz", ".jsonl.gz"):
        one = [f for f in qrels if f.endswith(ext)]
        if one:
            qrels = one[:1]
            break
    if not (corpus and queries and qrels):
        raise SystemExit(f"could not identify the files: corpus {corpus}, queries {queries}, qrels {qrels}")
    print("corpus:", corpus, "\nqueries:", queries, "\nqrels:", qrels)
    local = snapshot_download(repo, repo_type="dataset", allow_patterns=corpus + queries + qrels)
    os.makedirs(os.path.join(out, "qrels"), exist_ok=True)
    n = 0
    with open(os.path.join(out, "corpus.jsonl"), "w", encoding="utf-8") as f:
        for p in corpus:
            for r in rows(os.path.join(local, p)):
                f.write(json.dumps({"_id": str(r.get("_id", r.get("id"))), "title": r.get("title") or "",
                                    "text": r.get("text") or ""}, ensure_ascii=False) + "\n")
                n += 1
    m = 0
    with open(os.path.join(out, "queries.jsonl"), "w", encoding="utf-8") as f:
        for p in queries:
            for r in rows(os.path.join(local, p)):
                f.write(json.dumps({"_id": str(r.get("_id", r.get("id"))), "text": r.get("text") or ""}, ensure_ascii=False) + "\n")
                m += 1
    k = 0
    with open(os.path.join(out, "qrels", "test.tsv"), "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        seen = set()
        for p in qrels:
            for r in rows(os.path.join(local, p)):
                q = r.get("query-id", r.get("query_id", r.get("qid")))
                d = r.get("corpus-id", r.get("corpus_id", r.get("docid")))
                s = r.get("score", 1)
                if (q, d) in seen:
                    continue
                seen.add((q, d))
                f.write(f"{q}\t{d}\t{s}\n")
                k += 1
    print(f"wrote {out}: {n} records, {m} queries, {k} qrels")


if __name__ == "__main__":
    main()
