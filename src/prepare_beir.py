#!/usr/bin/env python3
"""
prepare_beir.py: turn a BEIR dataset (default DBpedia-Entity v2; --name scifact, nfcorpus, fiqa, ...) into the corpus.jsonl / queries.jsonl format of combo_search.

  corpus.jsonl  : {"id": doc_id, "fields": {"title": ..., "abstract": ...}}
  queries.jsonl : {"qid": ..., "query": ..., "answers": [doc ids with relevance >= 1], "qtype": ...}

qtype comes from the DBpedia-Entity v2 query id prefix (fixed in advance):
  SemSearch_ES                        -> "named entity"   (e.g. "brooklyn bridge")
  INEX_LD                             -> "keyword"        (e.g. "electronic music genres")
  INEX_XER, SemSearch_LS, TREC_Entity -> "list"           (e.g. "professional sports teams in Philadelphia")
  QALD2                               -> "question"       (e.g. "Who is the mayor of Berlin?")

Data source, in this order:
  1. --beir-dir DIR   an unzipped BEIR folder (corpus.jsonl, queries.jsonl, qrels/test.tsv)
                      e.g.  wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/dbpedia-entity.zip
  2. otherwise the Hugging Face datasets BeIR/dbpedia-entity and BeIR/dbpedia-entity-qrels

Usage:
  python prepare_beir.py --split test --out dbpedia_test
  python prepare_beir.py --split dev  --out dbpedia_dev --like dbpedia_test     # queries only, table linked
"""
import argparse
import csv
import gzip
import io
import json
import os

QTYPES = [("SemSearch_ES", "named entity"), ("INEX_LD", "keyword"), ("INEX_XER", "list"),
          ("SemSearch_LS", "list"), ("TREC_Entity", "list"), ("QALD2", "question")]


def qtype_of(qid):
    for pre, t in QTYPES:
        if str(qid).startswith(pre):
            return t
    return "other"


def open_any(path):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, encoding="utf-8")


def iter_jsonl_or_parquet(path):
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=50000):
            for row in batch.to_pylist():
                yield row
    else:
        with open_any(path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def hf_files(split, name="dbpedia-entity"):
    from huggingface_hub import hf_hub_download, list_repo_files
    main, qrel = f"BeIR/{name}", f"BeIR/{name}-qrels"
    files = list_repo_files(main, repo_type="dataset")
    qfiles = list_repo_files(qrel, repo_type="dataset")

    def pick(cands, key):
        c = [f for f in cands if key in os.path.basename(f).lower() or key in f.lower()]
        c.sort(key=lambda f: (not f.endswith((".jsonl.gz", ".jsonl")), len(f)))
        return c

    corpus = pick(files, "corpus")
    queries = pick(files, "queries")
    qr = [f for f in qfiles if os.path.basename(f).startswith(split) and f.endswith(".tsv")]
    if not (corpus and queries and qr):
        raise SystemExit(f"could not find files on Hugging Face.\n  {main}: {files}\n  {qrel}: {qfiles}\n"
                         "Download the BEIR zip instead and use --beir-dir.")
    dl = lambda repo, f: hf_hub_download(repo, f, repo_type="dataset")  # noqa: E731
    corpus_paths = [dl(main, f) for f in corpus if f.endswith(".parquet")] if corpus[0].endswith(".parquet") \
        else [dl(main, corpus[0])]
    query_paths = [dl(main, f) for f in queries if f.endswith(".parquet")] if queries[0].endswith(".parquet") \
        else [dl(main, queries[0])]
    return corpus_paths, query_paths, dl(qrel, qr[0])


def read_qrels(path):
    rel = {}
    with open(path, encoding="utf-8") as f:
        rd = csv.reader(f, delimiter="\t")
        for row in rd:
            if len(row) < 3 or row[0] in ("query-id", "qid"):
                continue
            q, d, s = row[0], row[1], row[-1]
            try:
                s = int(float(s))
            except ValueError:
                continue
            if s >= 1:
                rel.setdefault(q, []).append(d)
    return rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="dbpedia-entity",
                    help="BEIR dataset name, e.g. dbpedia-entity, scifact, nfcorpus, fiqa, arguana, scidocs")
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--beir-dir", default="", help="unzipped BEIR dbpedia-entity folder")
    ap.add_argument("--like", default="", help="existing folder: link its table/embeddings, write queries only")
    args = ap.parse_args()

    if args.beir_dir:
        d = args.beir_dir
        corpus_paths = [os.path.join(d, "corpus.jsonl")]
        query_paths = [os.path.join(d, "queries.jsonl")]
        qrel_path = os.path.join(d, "qrels", f"{args.split}.tsv")
    else:
        corpus_paths, query_paths, qrel_path = hf_files(args.split, args.name)

    rel = read_qrels(qrel_path)
    os.makedirs(args.out, exist_ok=True)

    if args.like:
        src = os.path.abspath(args.like)
        for name in os.listdir(src):
            if name == "corpus.jsonl" or name.startswith("rows_") or name.startswith("cols_"):
                dst = os.path.join(args.out, name)
                if not os.path.exists(dst):
                    os.symlink(os.path.join(src, name), dst)
        print(f"linked table and embeddings from {src}")
    else:
        n = 0
        with open(os.path.join(args.out, "corpus.jsonl"), "w", encoding="utf-8") as f:
            for p in corpus_paths:
                for r in iter_jsonl_or_parquet(p):
                    did = str(r.get("_id", r.get("id")))
                    fields = {}
                    if (r.get("title") or "").strip():
                        fields["title"] = r["title"].strip()
                    if (r.get("text") or "").strip():
                        fields["abstract"] = r["text"].strip()
                    f.write(json.dumps({"id": did, "fields": fields}, ensure_ascii=False) + "\n")
                    n += 1
        print(f"wrote {n} rows to {args.out}/corpus.jsonl")

    qtext = {}
    for p in query_paths:
        for r in iter_jsonl_or_parquet(p):
            qtext[str(r.get("_id", r.get("id")))] = r["text"]
    typed = args.name == "dbpedia-entity"
    queries = [{"qid": q, "query": qtext[q], "answers": ans, "qtype": qtype_of(q) if typed else ""}
               for q, ans in rel.items() if q in qtext]
    queries.sort(key=lambda x: x["qid"])
    with open(os.path.join(args.out, "queries.jsonl"), "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    counts = {}
    for q in queries:
        counts[q["qtype"]] = counts.get(q["qtype"], 0) + 1
    print(f"wrote {len(queries)} '{args.split}' queries; by type: {counts}")
    if typed and counts.get("other", 0) == len(queries):
        print("WARNING: no query id matched the DBpedia-Entity prefixes; check the ids, e.g.", queries[:2])


if __name__ == "__main__":
    main()
