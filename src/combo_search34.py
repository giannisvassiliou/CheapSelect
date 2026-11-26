#!/usr/bin/env python3
"""
combo_search.py — training-free multi-column search by "best column combination".

Idea
----
Pool     Whole-row BM25 and whole-row dense search each return their top-N rows;
         their union is the candidate pool (a common hybrid pool design).
Combo    For every pool row, combinations of 1..K of its columns are scored:
           cosine(query, mean of the column vectors)
           + w * keyword score (sum of per-column BM25, scaled per query)
           - small penalty per extra column.
         A row's score is its best combination. Optionally fused with RRF.
Optional A cross-encoder reads the RRF top-P rows twice: the full (truncated) row,
         and only each row's best combination.

Baselines run with the SAME models (no training anywhere): BM25, dense, RRF of
the two, and RRF over all 2C+2 lists (whole row + every column, both retrievers).
p-values: paired randomisation test against RRF(BM25+dense).

Per-column dense vectors are computed ONCE for all rows and cached on disk.
No labels are used anywhere; labels are only read to EVALUATE.

Subcommands
-----------
  make-toy       write a tiny synthetic dataset (smoke test)
  prepare-stark  convert a STaRK collection (amazon / mag / prime) to JSONL
  run            run stage 1, stage 2, optional cross-encoder, and evaluate

Data format (JSONL)
-------------------
  corpus.jsonl   {"id": "17", "fields": {"brand": "Adidas", "title": "...", ...}}
  queries.jsonl  {"qid": "q1", "query": "comfortable Adidas shoes", "answers": ["17"]}

Quick start
-----------
  pip install numpy sentence-transformers            # + stark-qa for STaRK
  python combo_search.py make-toy --out toy
  python combo_search.py run --data toy --encoder hash          # no downloads
  python combo_search.py prepare-stark --name amazon --out amz \
         --max-queries 300 --sample-rows 50000
  python combo_search.py run --data amz --encoder sentence-transformers/all-MiniLM-L6-v2 \
         --cross-encoder cross-encoder/ms-marco-MiniLM-L-6-v2

Caveat: --sample-rows shrinks the corpus (gold rows are always kept), which makes
the task easier. Absolute numbers are then NOT comparable to published full-corpus results;
only compare methods against each other on the same sample.
"""

import argparse
import collections
import itertools
import json
import os
import random
import re
import sys
import time
import zlib

import numpy as np


# ----------------------------------------------------------------------------- io

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# Metadata keys mFAR drops from Amazon review / Q&A records (mfar/data/format.py).
DROP_KEYS = {"reviewerID", "style", "verified", "overall", "reviewTime", "vote",
             "questionType", "answerType", "answerTime"}


def clean(value, max_chars=None):
    """Flatten lists/dicts to one line of text."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = " ; ".join(clean(v) for v in value if v is not None)
    elif isinstance(value, dict):
        value = " ; ".join(f"{k}: {clean(v)}" for k, v in value.items()
                           if v is not None and k not in DROP_KEYS)
    text = " ".join(str(value).split())
    return text[:max_chars] if max_chars else text



def safe_load(path):
    """Load a cached .npy file; return None if it is missing or damaged."""
    if not os.path.exists(path):
        return None
    try:
        return np.load(path)
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: damaged cache file {path} ({e}); it will be recomputed")
        return None


def safe_save(path, array):
    """Write to a temporary file first, then rename, so an interrupted save never
    leaves a damaged file behind."""
    tmp = path + ".tmp.npy"
    np.save(tmp, array)
    os.replace(tmp, path)

# ----------------------------------------------------------------------- encoders

def l2norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


class HashEncoder:
    """Bag-of-words hashing encoder. Only for smoke tests (no model download)."""

    name = "hash"

    def __init__(self, dim=2048):
        self.dim = dim

    def encode(self, texts, batch_size=None):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                tok = tok.strip(".,;:|!?()[]\"'")
                if tok:
                    out[i, zlib.crc32(tok.encode()) % self.dim] += 1.0
        return l2norm(out)


class STEncoder:
    """sentence-transformers bi-encoder (SBERT, MiniLM, BGE, E5, Contriever...)."""

    def __init__(self, name, device=None, max_seq_len=None, prefix=""):
        from sentence_transformers import SentenceTransformer
        self.name = name
        self.prefix = prefix
        self.model = SentenceTransformer(name, device=device)
        if max_seq_len:
            self.model.max_seq_length = max_seq_len

    def encode(self, texts, batch_size=64):
        texts = [self.prefix + t for t in texts]
        emb = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                normalize_embeddings=True,
                                show_progress_bar=len(texts) > 5000)
        return emb.astype(np.float32)


# -------------------------------------------------------------------------- text

def field_text(col, val, use_names):
    return f"{col}: {val}" if use_names else val


def row_text(fields, cols, use_names, table=None):
    parts = [field_text(c, fields[c], use_names) for c in cols if fields.get(c)]
    body = " | ".join(parts)
    return f"{table} | {body}" if (table and use_names) else body


# ----------------------------------------------------------------------- stage 1

def topk_cosine(Q, D, k, chunk=100_000):
    """Exact top-k by dot product (vectors are normalised), chunked over D."""
    k = min(k, D.shape[0])
    idx_parts, sc_parts = [], []
    for s in range(0, D.shape[0], chunk):
        S = Q @ D[s:s + chunk].T
        kk = min(k, S.shape[1])
        idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        idx_parts.append(idx + s)
        sc_parts.append(np.take_along_axis(S, idx, axis=1))
    I = np.concatenate(idx_parts, axis=1)
    Sc = np.concatenate(sc_parts, axis=1)
    order = np.argsort(-Sc, axis=1)[:, :k]
    return np.take_along_axis(I, order, axis=1), np.take_along_axis(Sc, order, axis=1)


# ------------------------------------------------------------------------ BM25

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = set("""a an and are as at be by can could do does for from has have how in is it its
of on or that the their them these this those to was were what which who whom whose with""".split())


def tokenize(text):
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


class BM25:
    """Okapi BM25 (k1=1.2, b=0.75) over a list of texts; one sparse lookup per query."""

    def __init__(self, texts, k1=1.2, b=0.75):
        from scipy import sparse
        self.vocab = {}
        rows, cols, vals = [], [], []
        lens = np.zeros(len(texts), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = tokenize(t)
            lens[i] = len(toks)
            cnt = {}
            for tok in toks:
                j = self.vocab.setdefault(tok, len(self.vocab))
                cnt[j] = cnt.get(j, 0) + 1
            for j, c in cnt.items():
                rows.append(i)
                cols.append(j)
                vals.append(c)
        n, V = len(texts), max(len(self.vocab), 1)
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        tf = np.asarray(vals, dtype=np.float32)
        df = np.bincount(cols, minlength=V).astype(np.float32)
        idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
        norm = k1 * (1 - b + b * lens / max(float(lens.mean()) if n else 1.0, 1e-9))
        w = idf[cols] * tf * (k1 + 1) / (tf + norm[rows])
        self.W = sparse.csc_matrix((w.astype(np.float32), (rows, cols)), shape=(n, V))
        self.n = n
        self.idf = idf.astype(np.float64)
        self.k1 = k1

    def score(self, qtoks):
        ids = sorted({self.vocab[t] for t in qtoks if t in self.vocab})
        if not ids:
            return np.zeros(self.n, dtype=np.float32)
        return np.asarray(self.W[:, ids].sum(axis=1)).ravel()


# ------------------------------------------------------------- per-column store

def safe_name(col):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", col) + f"_{zlib.crc32(col.encode()) & 0xFFFF:04x}"


class ColumnStore:
    """For every column: the rows that have a value, a BM25 index over those values,
    and dense vectors of "column: value" for ALL rows. Dense vectors are computed
    once and cached on disk, so later runs never re-embed."""

    def __init__(self, corpus, columns, encoder, args, cache_dir):
        N = len(corpus)
        self.cols = columns
        self.rows, self.pos, self.vecs, self.bm25 = {}, {}, {}, {}
        os.makedirs(cache_dir, exist_ok=True)
        for c in columns:
            rows = np.array([i for i, r in enumerate(corpus) if r["fields"].get(c)], dtype=np.int64)
            pos = np.full(N, -1, dtype=np.int64)
            pos[rows] = np.arange(len(rows))
            self.rows[c], self.pos[c] = rows, pos
            if len(rows) == 0:
                self.vecs[c], self.bm25[c] = None, None
                continue
            values = [corpus[i]["fields"][c] for i in rows]
            self.bm25[c] = BM25(values)
            f = os.path.join(cache_dir, safe_name(c) + ".npy")
            V = safe_load(f) if encoder.name != "hash" else None
            if V is None or V.shape[0] != len(rows):
                t1 = time.time()
                V = encoder.encode([field_text(c, v, args.use_names) for v in values],
                                   batch_size=args.batch_size)
                if encoder.name != "hash":
                    safe_save(f, V.astype(np.float16))
                print(f"  embedded column '{c}' ({len(rows)} values, {time.time() - t1:.0f}s)")
            self.vecs[c] = np.ascontiguousarray(V, dtype=np.float32)


# ------------------------------------------------------------------ ranking

def top_idx(scores, k):
    k = min(k, len(scores))
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def rrf(lists, k=60):
    out = {}
    for L in lists:
        for rank, r in enumerate(L, start=1):
            out[int(r)] = out.get(int(r), 0.0) + 1.0 / (k + rank)
    return out


def ranked_by(score_dict):
    return [r for r, _ in sorted(score_dict.items(), key=lambda x: -x[1])]


def combos_of(items, kmax):
    out = []
    for r in range(1, min(kmax, len(items)) + 1):
        out.extend(itertools.combinations(items, r))
    return out


def combo_scores(pool, store, col_dense, col_bm, q, weights, args):
    """Best column combination per pool row, for each keyword weight w.
    Combination score = cosine(query, mean of column vectors)
                        + w * (sum of column BM25 scores, scaled to [0,1] per query)
                        - size_penalty * (columns - 1)."""
    singles, max_single_bm = [], 0.0
    for r in pool:
        items = []
        for c in store.cols:
            p = store.pos[c][r]
            if p >= 0:
                items.append((c, int(p), float(col_dense[c][p]), float(col_bm[c][p])))
        singles.append(items)
        if items:
            max_single_bm = max(max_single_bm, max(x[3] for x in items))
    ms = max_single_bm or 1.0
    wsel = max(weights)
    cands, orders, max_combo_bm = [], [], 0.0
    for items in singles:
        if not items:
            cands.append([])
            orders.append([])
            continue
        items = sorted(items, key=lambda x: -(x[2] + wsel * x[3] / ms))
        orders.append([x[0] for x in items])  # all columns, most query-relevant first
        items = items[: args.prefilter_cols]
        out = []
        for cb in combos_of(items, args.max_combo):
            if len(cb) == 1:
                cos = cb[0][2]
            else:
                v = sum(store.vecs[c][p] for c, p, _, _ in cb)
                cos = float(v @ q / (np.linalg.norm(v) + 1e-9))
            bm = sum(x[3] for x in cb)
            max_combo_bm = max(max_combo_bm, bm)
            out.append((tuple(x[0] for x in cb), cos, bm))
        cands.append(out)
    mb = max_combo_bm or 1.0
    results = {}
    for w in weights:
        sc = np.full(len(pool), -1e9, dtype=np.float32)
        best = []
        for i, out in enumerate(cands):
            if not out:
                best.append(())
                continue
            vals = [cos + w * bm / mb - args.size_penalty * (len(cb) - 1) for cb, cos, bm in out]
            j = int(np.argmax(vals))
            sc[i] = vals[j]
            best.append(out[j][0])
        results[w] = (sc, best)
    return results, orders


def combo_scores_exact(pool, store, col_dense, col_bm, q, bw, args, corpus, encoder):
    """Same as combo_scores (hybrid, weight bw), but a multi-column combination is
    scored by embedding its JOINED text instead of averaging its column vectors."""
    singles, max_single_bm = [], 0.0
    for r in pool:
        items = []
        for c in store.cols:
            p = store.pos[c][r]
            if p >= 0:
                items.append((c, int(p), float(col_dense[c][p]), float(col_bm[c][p])))
        singles.append(items)
        if items:
            max_single_bm = max(max_single_bm, max(x[3] for x in items))
    ms = max_single_bm or 1.0
    cands, need = [], []
    for r, items in zip(pool, singles):
        if not items:
            cands.append([])
            continue
        items = sorted(items, key=lambda x: -(x[2] + bw * x[3] / ms))[: args.prefilter_cols]
        out = []
        for cb in combos_of(items, args.max_combo):
            cols = tuple(x[0] for x in cb)
            out.append([cols, cb[0][2] if len(cb) == 1 else None, sum(x[3] for x in cb)])
            if len(cb) > 1:
                need.append((len(cands), len(out) - 1, row_text(corpus[r]["fields"], cols, args.use_names)))
        cands.append(out)
    if need:
        V = encoder.encode([t for _, _, t in need], batch_size=args.batch_size)
        for (i, j, _), v in zip(need, V):
            cands[i][j][1] = float(v @ q)
    mb = max((cb[2] for out in cands for cb in out), default=0.0) or 1.0
    sc = np.full(len(pool), -1e9, dtype=np.float32)
    best = []
    for i, out in enumerate(cands):
        if not out:
            best.append(())
            continue
        vals = [cos + bw * bm / mb - args.size_penalty * (len(cols) - 1) for cols, cos, bm in out]
        j = int(np.argmax(vals))
        sc[i] = vals[j]
        best.append(out[j][0])
    return sc, best


# ------------------------------------------------------- confidence predictors

PREDICTORS = ("margin top1-top2", "top5 vs rest", "NQC-like spread", "WIG-like",
              "term coverage top1", "BM25-dense agreement", "BM25-combo agreement")


def confidence_predictors(s_bm_all, L_bm, L_dense, rank_combo, qt, corpus, columns):
    """Training-free signals of how much to trust BM25 for this question (higher = trust more)."""
    sc = np.asarray(s_bm_all[L_bm], dtype=np.float64)
    s1 = sc[0] if len(sc) else 0.0
    out = {}
    out["margin top1-top2"] = (s1 - sc[1]) / s1 if len(sc) > 1 and s1 > 0 else 0.0
    out["top5 vs rest"] = ((sc[:5].mean() - sc[19:].mean()) / s1) if len(sc) > 20 and s1 > 0 else 0.0
    out["NQC-like spread"] = (sc.std() / sc.mean()) if len(sc) and sc.mean() > 0 else 0.0
    nq = max(len(set(qt)), 1)
    out["WIG-like"] = (sc[:5].mean() - float(np.mean(s_bm_all))) / np.sqrt(nq) if len(sc) else 0.0
    qset = set(qt)
    if len(L_bm) and qset:
        top_text = " ".join(corpus[int(L_bm[0])]["fields"].get(c, "") for c in columns)
        out["term coverage top1"] = len(qset & set(tokenize(top_text))) / len(qset)
    else:
        out["term coverage top1"] = 0.0
    a, b, c10 = set(map(int, L_bm[:10])), set(map(int, L_dense[:10])), set(rank_combo[:10])
    out["BM25-dense agreement"] = len(a & b) / 10.0
    out["BM25-combo agreement"] = len(a & c10) / 10.0
    return out


CN_DECAY = (0.8, 0.6, 0.4, 0.2, 0.1, 0.05)  # fixed in advance
CN_NAMES = "COMBO-NEW names   avg-zero"
CN_CONTENT = "COMBO-NEW content avg-zero"
CN_VARIANTS = (  # (label, relevance source, mode); avg-zero = the main version, fixed in advance
    ("COMBO-NEW names   avg-zero", "names", "zero"),
    ("COMBO-NEW content avg-zero", "content", "zero"),
    ("COMBO-NEW content avg-renorm", "content", "renorm"),
    ("COMBO-NEW content decay", "content", "decay"),
    ("COMBO-NEW names   decay", "names", "decay"),
)


def nice_name(col):
    return re.sub(r"[_.]+", " ", col).strip()


def combo_new_scores(rel, store, col_dense, N, k, mode="decay"):
    """rel: {column: relevance to the question}. The k most relevant columns are used for EVERY row.
    mode 'decay' : sum_i decay_i * rel_i * sim_i                      (original proposal)
    mode 'zero'  : sum_i rel_i * sim_i / sum_i rel_i  (missing value = similarity 0)
    mode 'renorm': sum over PRESENT columns of rel_i * sim_i / sum over PRESENT columns of rel_i"""
    cols = sorted(rel, key=lambda c: -rel[c])[:k]
    num = np.zeros(N, dtype=np.float64)
    den = np.zeros(N, dtype=np.float64)
    tot = 0.0
    for i, c in enumerate(cols):
        r = max(rel[c], 0.0)
        w = (CN_DECAY[i] if i < len(CN_DECAY) else CN_DECAY[-1]) * r if mode == "decay" else r
        num[store.rows[c]] += w * col_dense[c]
        den[store.rows[c]] += w
        tot += w
    if mode == "decay":
        return num, cols
    if mode == "zero":
        return num / (tot if tot > 0 else 1.0), cols
    S = np.full(N, -1e9)
    ok = den > 0
    S[ok] = num[ok] / den[ok]
    return S, cols


# ---------------------------------------------------------------- GLiNER + BM25
# Label -> columns where that value should appear. Fixed in advance, per dataset.
# A dataset map is used only for the columns that exist in the run.
GL_MAPS = {
    "amazon": {
        "brand": ["brand"],
        "product type": ["title"],
        "material": ["feature", "description", "title"],
        "color": ["feature", "description", "title"],
        "product feature": ["feature", "description", "title"],
    },
    "mag": {
        "person": ["author___affiliated_with___institution"],
        "organization": ["author___affiliated_with___institution"],
        "research topic": ["paper___has_topic___field_of_study", "title", "abstract"],
    },
    "dbpedia": {  # DBpedia-Entity: a named thing should appear in the entity's name or its abstract
        "person": ["title", "abstract"], "organization": ["title", "abstract"],
        "location": ["title", "abstract"], "creative work": ["title", "abstract"],
        "event": ["title", "abstract"], "product": ["title", "abstract"],
    },
    "prime": {  # entity names live in the name column and in the relation (link) columns
        "drug": None, "disease": None, "gene or protein": None,
        "symptom or phenotype": None, "anatomy": None, "biological pathway": None,
    },
}
NEG_WORDS = {"not", "no", "without", "except", "excluding", "nt", "isn", "aren", "doesn", "don", "never"}
GL_MAIN = "GLINER+BM25 (lambda=1.0)"
GL_LINES = []
GL_FILTER = "  [design] GLiNER hard filter"
GL_PENALTY = "  [design] GLiNER penalty-only"
GL_GATED = "  [design] GLiNER gated (top-1 viol.)"


def detect_gl_dataset(columns):
    cs = set(columns)
    if "brand" in cs:
        return "amazon"
    if any(c.startswith("author___") for c in cs):
        return "mag"
    if "indication" in cs or "target" in cs:
        return "prime"
    if cs == {"title", "abstract"}:
        return "dbpedia"
    return None


def resolve_gl_map(name, columns, path=""):
    if path:
        with open(path) as f:
            raw = json.load(f)
    else:
        raw = GL_MAPS[name]
    out = {}
    for lab, cols in raw.items():
        if cols is None:  # prime: name + every non-"details" column
            cols = [c for c in columns if not c.startswith("details")]
        cols = [c for c in cols if c in columns]
        if cols:
            out[lab] = cols
    return out


def gliner_spans(qtexts, labels, args, cache_file):
    """Tag every question once and cache the result (spans with label, score, negation flag)."""
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return [json.loads(l)["spans"] for l in f]
    from gliner import GLiNER
    model = GLiNER.from_pretrained(args.gl_model)
    try:
        model = model.to(args.device or "cpu")
    except Exception:  # noqa: BLE001
        pass
    out = []
    for t in qtexts:
        ents = model.predict_entities(t, labels, threshold=args.gl_threshold)
        spans = []
        for e in ents:
            before = tokenize_keep(t[: int(e.get("start", 0))])[-3:]
            spans.append({"text": e["text"], "label": e["label"], "score": float(e["score"]),
                          "neg": any(w in NEG_WORDS for w in before)})
        out.append(spans)
    save_jsonl(cache_file, [{"query": t, "spans": sp} for t, sp in zip(qtexts, out)])
    return out


def tokenize_keep(text):
    """Like tokenize, but keeps stop/negation words (used only for the negation check)."""
    return _TOKEN_RE.findall(text.lower())


def gliner_bonus(spans, cand, corpus, gl_map):
    """bonus(row) = sum over tagged spans of score * (share of the span's words found in the
    span's own columns); negated spans give a penalty instead of a bonus."""
    b = np.zeros(len(cand), dtype=np.float64)
    used = [sp for sp in spans if sp["label"] in gl_map and tokenize(sp["text"])]
    if not used:
        return b, 0
    cache = {}
    for i, r in enumerate(cand):
        f = corpus[int(r)]["fields"]
        for sp in used:
            cols = tuple(gl_map[sp["label"]])
            if cols not in cache:
                cache[cols] = {}
            if r not in cache[cols]:
                cache[cols][r] = set(tokenize(" ".join(f.get(c, "") for c in cols)))
            toks = tokenize(sp["text"])
            match = sum(t in cache[cols][r] for t in toks) / len(toks)
            b[i] += (-1.0 if sp["neg"] else 1.0) * sp["score"] * match
    return b, len(used)


def gliner_match_matrix(spans, cand, corpus, gl_map):
    """M[i, j] = share of span j's words found in span j's own columns of candidate i."""
    used = [sp for sp in spans if sp["label"] in gl_map and tokenize(sp["text"])]
    M = np.zeros((len(cand), len(used)))
    for i, r in enumerate(cand):
        f = corpus[int(r)]["fields"]
        for j, sp in enumerate(used):
            toks = set(tokenize(" ".join(f.get(c, "") for c in gl_map[sp["label"]])))
            words = tokenize(sp["text"])
            M[i, j] = sum(t in toks for t in words) / len(words)
    return M, used


def crude_stem(t):
    """Very light stemming: plural and -ing/-ed endings (shoes->shoe, running->runn, lighter stays)."""
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 5 and t.endswith("ing"):
        return t[:-3]
    if len(t) > 4 and t.endswith("ed"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def gliner_match_stem(spans_used, cand, corpus, gl_map):
    M = np.zeros((len(cand), len(spans_used)))
    for i, r in enumerate(cand):
        f = corpus[int(r)]["fields"]
        for j, sp in enumerate(spans_used):
            toks = {crude_stem(t) for t in tokenize(" ".join(f.get(c, "") for c in gl_map[sp["label"]]))}
            words = [crude_stem(t) for t in tokenize(sp["text"])]
            M[i, j] = sum(t in toks for t in words) / len(words)
    return M


def gliner_match_sbert(spans_used, cand, store, gl_map, encoder, use_names):
    """S[i, j] = max over span j's columns of cos('col: span', 'col: value of row i')."""
    S = np.full((len(cand), len(spans_used)), -1.0)
    cand = np.asarray(cand, dtype=np.int64)
    for j, sp in enumerate(spans_used):
        cols = [c for c in gl_map[sp["label"]] if c in store.vecs and store.vecs[c] is not None]
        if not cols:
            continue
        qv = encoder.encode([field_text(c, sp["text"], use_names) for c in cols])
        for c, v in zip(cols, qv):
            pos = store.pos[c][cand]
            ok = pos >= 0
            if ok.any():
                S[ok, j] = np.maximum(S[ok, j], store.vecs[c][pos[ok]] @ v)
    return S


def gl_satisfies_soft(P, used):
    """P[i, j] = True if tag j counts as PRESENT in row i."""
    ok = np.ones(P.shape[0], bool)
    for j, sp in enumerate(used):
        ok &= ~P[:, j] if sp["neg"] else P[:, j]
    return ok


def gl_satisfies(M, used):
    """A row satisfies the tags if every positive tag is fully present and no negated tag is."""
    if not used:
        return np.ones(M.shape[0], bool)
    ok = np.ones(M.shape[0], bool)
    for j, sp in enumerate(used):
        ok &= (M[:, j] < 1.0) if sp["neg"] else (M[:, j] >= 1.0)
    return ok


FAILURE_SIGNALS = ("IDF coverage top1", "rare-term coverage top10", "known-word rate",
                   "absolute BM25 strength", "query specificity")
MAIN_ASYM = ("IDF coverage top1", 10)  # fixed in advance: switch away from BM25 for the 10% lowest


def failure_signals(row_bm, s_bm_all, L_bm, qt, corpus, columns):
    """Absolute signs that BM25 FAILED (higher = BM25 more trustworthy)."""
    q = list(dict.fromkeys(qt))
    known = [t for t in q if t in row_bm.vocab]
    idf = {t: row_bm.idf[row_bm.vocab[t]] for t in known}
    out = {"known-word rate": len(known) / len(q) if q else 0.0}

    def row_tokens(r):
        return set(tokenize(" ".join(corpus[int(r)]["fields"].get(c, "") for c in columns)))

    tot = sum(idf.values())
    if len(L_bm) and tot > 0:
        t1 = row_tokens(L_bm[0])
        out["IDF coverage top1"] = sum(v for t, v in idf.items() if t in t1) / tot
        rare = sorted(idf, key=lambda t: -idf[t])[:3]
        top10 = set().union(*(row_tokens(r) for r in L_bm[:10]))
        out["rare-term coverage top10"] = sum(t in top10 for t in rare) / len(rare)
        out["absolute BM25 strength"] = float(s_bm_all[L_bm[0]]) / (tot * (row_bm.k1 + 1))
        out["query specificity"] = float(np.mean(list(idf.values()))) / float(row_bm.idf.max())
    else:
        for k in ("IDF coverage top1", "rare-term coverage top10", "absolute BM25 strength", "query specificity"):
            out[k] = 0.0
    return out


def report_asymmetric(rows, perm, title="combo"):
    """Keep BM25 by default; switch to the alternative only for the X% most suspicious questions."""
    keys = ("Hit@1", "Hit@5", "R@20", "MRR")
    bm = {k: np.array([r["bm"][k] for r in rows]) for k in keys}
    cb = {k: np.array([r["combo"][k] for r in rows]) for k in keys}
    n = len(rows)
    bad = bm["Hit@5"] == 0
    rescue = bad & (cb["Hit@1"] == 1)
    print(f"\n=== Asymmetric gate: BM25 by default, switch to [{title}] only when BM25 looks like it FAILED ===")
    print(f"BM25 has nothing correct in its top 5 for {int(bad.sum())}/{n} questions; "
          f"[{title}] has the right answer first in {int(rescue.sum())} of them.")
    print(f"CEILING of a perfect asymmetric gate: Hit@1 {100 * bm['Hit@1'].mean():.1f} -> "
          f"{100 * np.where(rescue, 1, bm['Hit@1']).mean():.1f}  (+{100 * rescue.mean():.1f})\n")
    names = FAILURE_SIGNALS + ("margin top1-top2", "top5 vs rest", "WIG-like")
    hdr = f"{'signal':26s} {'AUC':>6s}  " + "  ".join(f"{'switch ' + str(x) + '%':>20s}" for x in (5, 10, 15, 20))
    print(hdr)
    print(f"{'':26s} {'':6s}  " + "  ".join(f"{'H@1 / MRR (p H@1)':>20s}" for _ in range(4)))
    diff = bm["MRR"] != cb["MRR"]
    lab = bm["MRR"] > cb["MRR"]
    out = {}
    for name in names:
        if name not in rows[0]["pred"]:
            continue
        v = np.array([r["pred"][name] for r in rows])
        a = auc(v[diff], lab[diff])
        cells, res = [], {}
        for x in (5, 10, 15, 20):
            k = max(1, int(round(n * x / 100)))
            switch = np.zeros(n, bool)
            switch[np.argsort(v, kind="mergesort")[:k]] = True  # the k least trustworthy for BM25
            m = {kk: np.where(switch, cb[kk], bm[kk]) for kk in keys}
            res[x] = {kk: 100 * float(m[kk].mean()) for kk in keys}
            res[x]["p_H1_vs_BM25"] = perm_test(m["Hit@1"], bm["Hit@1"], perm)
            res[x]["p_MRR_vs_BM25"] = perm_test(m["MRR"], bm["MRR"], perm)
            mark = "*" if (name, x) == MAIN_ASYM else " "
            cells.append(f"{res[x]['Hit@1']:5.1f} / {res[x]['MRR']:5.1f} ({res[x]['p_H1_vs_BM25']:.2f}){mark}")
        out[name] = {"auc": a, "gates": res}
        print(f"{name:26s} {a:6.3f}  " + "  ".join(f"{c:>20s}" for c in cells))
    print(f"\nBM25 alone: H@1 {100 * bm['Hit@1'].mean():.1f}  MRR {100 * bm['MRR'].mean():.1f}"
          f"     * = the policy fixed in advance ({MAIN_ASYM[0]}, switch {MAIN_ASYM[1]}%)")
    return out


def auc(score, label):
    """Probability that a random positive gets a higher score than a random negative."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):  # average ranks for ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def report_predictors(rows, perm):
    """rows: list of dicts with 'pred', 'bm' (metrics), 'combo' (metrics)."""
    keys = ("Hit@1", "Hit@5", "R@20", "MRR")
    bm = {k: np.array([r["bm"][k] for r in rows]) for k in keys}
    cb = {k: np.array([r["combo"][k] for r in rows]) for k in keys}
    diff = bm["MRR"] != cb["MRR"]
    lab = bm["MRR"] > cb["MRR"]
    print("\n=== Confidence predictors for the per-question gate ===")
    print(f"questions where BM25 and combo differ (MRR): {int(diff.sum())} of {len(rows)};"
          f" BM25 better on {int((lab & diff).sum())}, combo better on {int((~lab & diff).sum())}")
    print("AUC = how well the signal predicts 'BM25 better than combo' (0.5 = useless, 1.0 = perfect)\n")
    print(f"{'predictor':24s} {'AUC':>6s}   " + "  ".join(f"{'BM25 for ' + str(x) + '%':>14s}" for x in (10, 25, 50, 75, 90)))
    print(f"{'':24s} {'':6s}   " + "  ".join(f"{'H@1 / MRR':>14s}" for _ in range(5)))
    out = {}
    for name in PREDICTORS:
        v = np.array([r["pred"][name] for r in rows])
        a = auc(v[diff], lab[diff])
        cells, res = [], {}
        for x in (10, 25, 50, 75, 90):
            thr = np.quantile(v, 1 - x / 100)
            use_bm = v >= thr
            m = {k: np.where(use_bm, bm[k], cb[k]) for k in keys}
            res[x] = {k: 100 * float(m[k].mean()) for k in keys}
            res[x]["p_H1_vs_BM25"] = perm_test(m["Hit@1"], bm["Hit@1"], perm)
            res[x]["p_MRR_vs_BM25"] = perm_test(m["MRR"], bm["MRR"], perm)
            cells.append(f"{res[x]['Hit@1']:5.1f} / {res[x]['MRR']:5.1f}")
        out[name] = {"auc": a, "gates": res}
        print(f"{name:24s} {a:6.3f}   " + "  ".join(f"{c:>14s}" for c in cells))
    ora = {k: np.where(bm["MRR"] >= cb["MRR"], bm[k], cb[k]) for k in keys}
    print(f"\n{'BM25 alone':24s} {'':6s}   H@1 {100 * bm['Hit@1'].mean():5.1f}  H@5 {100 * bm['Hit@5'].mean():5.1f}"
          f"  R@20 {100 * bm['R@20'].mean():5.1f}  MRR {100 * bm['MRR'].mean():5.1f}")
    print(f"{'combo alone':24s} {'':6s}   H@1 {100 * cb['Hit@1'].mean():5.1f}  H@5 {100 * cb['Hit@5'].mean():5.1f}"
          f"  R@20 {100 * cb['R@20'].mean():5.1f}  MRR {100 * cb['MRR'].mean():5.1f}")
    print(f"{'ORACLE gate (uses labels)':24s} {'':6s}   H@1 {100 * ora['Hit@1'].mean():5.1f}  H@5 {100 * ora['Hit@5'].mean():5.1f}"
          f"  R@20 {100 * ora['R@20'].mean():5.1f}  MRR {100 * ora['MRR'].mean():5.1f}")
    print("(oracle = best possible gate; a real gate can only approach it)")
    return out


# --------------------------------------------------------------------- metrics

def metrics(ranked, gold):
    gold = set(gold)
    first = next((i for i, r in enumerate(ranked) if r in gold), None)
    return {
        "Hit@1": float(first == 0),
        "Hit@5": float(first is not None and first < 5),
        "R@20": len(gold & set(ranked[:20])) / max(len(gold), 1),
        "MRR": 0.0 if first is None else 1.0 / (first + 1),
    }


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-12 else x - x.mean()


def perm_test(a, b, n=10000, seed=0):
    """Two-sided paired randomisation (sign-flip) test on per-query values."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if not np.any(d):
        return 1.0
    rng = np.random.default_rng(seed)
    null = np.abs((rng.choice([-1.0, 1.0], size=(n, len(d))) * d).mean(axis=1))
    return float((np.sum(null >= abs(d.mean()) - 1e-12) + 1) / (n + 1))


# ------------------------------------------------------------------------- run

def build_encoder(name, args, prefix=""):
    if name == "hash":
        return HashEncoder()
    return STEncoder(name, device=args.device, max_seq_len=args.max_seq_len, prefix=prefix)


def row_label(fields, n=70):
    for k in ("title", "name"):
        if fields.get(k):
            return fields[k][:n]
    for v in fields.values():
        return v[:n]
    return "?"


def explain_row(r, corpus, store, col_dense, col_bm, mbm, bw, best, pos_rank):
    """Per-column evidence for one row (diagnostic mode)."""
    cols = []
    for c in store.cols:
        p = store.pos[c][r]
        if p >= 0:
            cols.append((c, float(col_dense[c][p]), float(col_bm[c][p]),
                         float(col_dense[c][p]) + bw * float(col_bm[c][p]) / mbm))
    cols.sort(key=lambda x: -x[3])
    out = [f"  - row {corpus[r]['id']}: **{row_label(corpus[r]['fields'])}**  "
           f"(rank by your method: {pos_rank.get(r, 'not in pool')})",
           f"    - chosen combination: {list(best.get(r, ())) or 'n/a'}"]
    for c, cos, bm, tot in cols[:6]:
        txt = corpus[r]["fields"][c][:90].replace("|", "/")
        out.append(f"    - `{c}` cos={cos:.3f} bm25={bm:.1f} score={tot:.3f} :: {txt}")
    return out


CE_FULL = "CE full row         (RRF top-P)"
CE_COMBO = "CE best combo only  (RRF top-P)"
CE_ORDER = "CE query-ordered row(RRF top-P)"
CE_FULL_MIX = "CE full row + combo (no training)"
CE_ORDER_MIX = "CE ordered  + combo (no training)"
RRF3 = "RRF  BM25+dense+column filter"
COMBO_BM25 = "combo + BM25 base   [post-hoc]"
COMBO_EXACT = "combo EXACT joined text [ablation]"
GATE_POS = "GATE per-position"
GATE_Q = "GATE per-question"
SWEEP_TAU = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
SWEEP_MU = (0.05, 0.1, 0.2, 0.3)
COMBO_PLUS = "combo hybrid  POOL+ (frozen)"
CE_PLUS_FULL = "CE full row   POOL+ (top-P by combo)"
CE_PLUS_MIX = "CE ordered+combo POOL+ (top-P combo)"


def cmd_run(args):
    t0 = time.time()
    corpus = load_jsonl(os.path.join(args.data, "corpus.jsonl"))
    queries = load_jsonl(os.path.join(args.data, "queries.jsonl"))
    if args.max_queries:
        random.Random(args.seed).shuffle(queries)
        queries = queries[: args.max_queries]
    id2idx = {str(r["id"]): i for i, r in enumerate(corpus)}
    queries = [q for q in queries if any(str(a) in id2idx for a in q["answers"])]

    columns = args.columns.split(",") if args.columns else sorted(
        {c for r in corpus for c in r["fields"]})
    missing = [c for c in columns if not any(r["fields"].get(c) for r in corpus[:: max(1, len(corpus) // 20000)])]
    if missing:
        print(f"WARNING: these columns look empty or misspelled: {missing}")
    if args.short_first:
        sample = corpus[:: max(1, len(corpus) // 5000)]
        med = {c: float(np.median([len(r["fields"][c]) for r in sample if r["fields"].get(c)] or [1e9]))
               for c in columns}
        columns = sorted(columns, key=lambda c: med[c])
    print(f"rows={len(corpus)}  queries={len(queries)}  columns={columns}")

    if args.bm25_only:
        # quick data check: whole-row BM25 only (no embeddings, no GPU)
        row_bm = BM25([" ".join(r["fields"][c] for c in columns if r["fields"].get(c)) for r in corpus])
        print(f"whole-row BM25 ready ({time.time() - t0:.0f}s)")
        ms = []
        for qrec in queries:
            gold = [id2idx[str(a)] for a in qrec["answers"] if str(a) in id2idx]
            ms.append(metrics([int(r) for r in top_idx(row_bm.score(tokenize(qrec["query"])), args.top_n)], gold))
        avg = {k: 100 * float(np.mean([m[k] for m in ms])) for k in ms[0]}
        print(f"\nBM25 (whole row) over {len(queries)} queries: "
              + "  ".join(f"{k} {v:.1f}" for k, v in avg.items()))
        return

    doc_enc = build_encoder(args.encoder, args, prefix=args.doc_prefix)
    q_enc = build_encoder(args.encoder, args, prefix=args.query_prefix) \
        if (args.query_prefix != args.doc_prefix and args.encoder != "hash") else doc_enc
    model_tag = f"{os.path.basename(args.encoder.rstrip('/'))}_{'names' if args.use_names else 'vals'}_{len(corpus)}"

    # ---- whole-row dense (cached)
    col_hash = zlib.crc32(("|".join(columns) + "#" + args.table_name).encode()) & 0xFFFFFFFF
    cache = os.path.join(args.data, f"rows_{model_tag}_{col_hash:08x}.npy")
    D = safe_load(cache) if args.encoder != "hash" else None
    if D is not None and D.shape[0] == len(corpus):
        D = D.astype(np.float32)
        print(f"loaded row embeddings from {cache}")
    else:
        D = doc_enc.encode([row_text(r["fields"], columns, args.use_names, args.table_name) for r in corpus],
                           batch_size=args.batch_size)
        if args.encoder != "hash":
            safe_save(cache, D.astype(np.float16))
    print(f"whole-row embeddings ready ({time.time() - t0:.0f}s)")

    # ---- whole-row BM25 (all columns, never truncated)
    row_bm = BM25([" ".join(r["fields"][c] for c in columns if r["fields"].get(c)) for r in corpus])
    print(f"whole-row BM25 ready ({time.time() - t0:.0f}s)")

    # ---- per-column BM25 + dense vectors for ALL rows (dense cached once)
    store = ColumnStore(corpus, columns, doc_enc, args, os.path.join(args.data, f"cols_{model_tag}"))
    store.cols = [c for c in columns if store.vecs[c] is not None]
    print(f"per-column indexes ready ({time.time() - t0:.0f}s)")
    if args.combo_new:
        name_vecs = dict(zip(store.cols, doc_enc.encode([nice_name(c) for c in store.cols],
                                                         batch_size=args.batch_size)))
        centroids = {c: l2norm(store.vecs[c].mean(axis=0, keepdims=True))[0] for c in store.cols}
        print("COMBO-NEW column relevance, example (names):",
              ", ".join(nice_name(c) for c in store.cols[:5]), "...")

    Q = q_enc.encode([q["query"] for q in queries], batch_size=args.batch_size)
    gl_all = None
    if args.gliner:
        gl_name = args.gl_dataset if args.gl_dataset != "auto" else detect_gl_dataset(columns)
        if not gl_name and not args.gl_map:
            raise SystemExit("--gliner: could not detect the dataset; pass --gl-dataset or --gl-map")
        gl_map = resolve_gl_map(gl_name, columns, args.gl_map)
        print(f"GLiNER label -> columns ({gl_name}): {gl_map}")
        tagfile = os.path.join(args.data, f"gliner_spans_{os.path.basename(args.gl_model.rstrip('/'))}"
                                           f"_{zlib.crc32('|'.join(sorted(gl_map)).encode()) & 0xFFFF:04x}"
                                           f"_{args.gl_threshold}_{len(queries)}_{args.seed}.jsonl")
        gl_all = gliner_spans([q["query"] for q in queries], list(gl_map), args, tagfile)
        tagged = sum(1 for sp in gl_all if any(x["label"] in gl_map for x in sp))
        lab_cnt = collections.Counter(x["label"] for sp in gl_all for x in sp)
        negs = sum(x["neg"] for sp in gl_all for x in sp)
        print(f"GLiNER: {tagged}/{len(queries)} questions have at least one tag; labels {dict(lab_cnt)};"
              f" negated spans {negs}  (tags cached in {tagfile})")
        for qq, sp in list(zip(queries, gl_all))[:5]:
            print(f"  e.g. {qq['query'][:90]!r} -> "
                  + ", ".join(f"{x['text']}={x['label']}{' [NOT]' if x['neg'] else ''}" for x in sp))
    K, bw = args.top_n, args.bm25_weight
    alphas = [float(x) for x in args.alphas.split(",")]
    ce = None
    if args.cross_encoder:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(args.cross_encoder, max_length=args.max_seq_len or 512, device=args.device)
        if args.ce_fp16:
            ce.model.half()
        print(f"cross-encoder '{args.cross_encoder}' on {ce.model.device}"
              f"{' (fp16)' if args.ce_fp16 else ''}; {2 * args.ce_budget} reads per query")

    REF = "RRF  BM25+dense (whole row)"
    res = {"BM25 (whole row)": [], "Dense (whole row)": [], REF: [],
           "RRF  all columns (2C+2 lists)": []}
    for al in alphas:
        res[f"combo dense    (alpha={al})"] = []
    for al in alphas:
        res[f"combo hybrid   (alpha={al})"] = []
    if ce:
        for key in (CE_FULL, CE_COMBO, CE_ORDER, CE_FULL_MIX, CE_ORDER_MIX):
            res[key] = []
    res[COMBO_BM25] = []
    global GL_LINES
    GL_LINES = [(1.0, GL_MAIN)]
    if args.gl_sweep:
        GL_LINES += [(0.5, "  [sens] GLINER+BM25 lambda=0.5"), (2.0, "  [sens] GLINER+BM25 lambda=2.0")]
    gl_used = []
    gl_ceil = []
    gl_mats = []
    if args.gliner and args.gl_ceiling:
        res[GL_FILTER] = []
        res[GL_PENALTY] = []
        res[GL_GATED] = []
    if args.gliner:
        for _, lab in GL_LINES:
            res[lab] = []
    if args.combo_new:
        for lab, _, _ in CN_VARIANTS:
            res[lab] = []
        pred_rows_cn = {CN_NAMES: [], CN_CONTENT: []}
    global GATE_POS, GATE_Q
    GATE_POS = f"GATE per-position (tau={args.gate_tau})"
    GATE_Q = f"GATE per-question (margin={args.gate_margin})"
    res[GATE_POS] = []
    res[GATE_Q] = []
    gate_kept, gate_bm_used = [], []
    pred_rows = []
    if args.gate_sweep:
        for t in SWEEP_TAU:
            res[f"  [sens] gate-pos tau={t}"] = []
        for m in SWEEP_MU:
            res[f"  [sens] gate-q margin={m}"] = []
    if args.exact_ablation:
        res[COMBO_EXACT] = []
    if args.col_filter:
        res[RRF3] = []
        res[COMBO_PLUS] = []
        if ce:
            res[CE_PLUS_FULL] = []
            res[CE_PLUS_MIX] = []
    ceiling, ceiling2, runs = [], [], []
    diag = []

    TIM = collections.defaultdict(list)
    for qi, qrec in enumerate(queries):
        gold = [id2idx[str(a)] for a in qrec["answers"] if str(a) in id2idx]
        q, qt = Q[qi], tokenize(qrec["query"])

        _t = time.perf_counter()
        L_dense = top_idx(D @ q, K)
        TIM["dense search (whole row, vectors ready)"].append(time.perf_counter() - _t)
        _t = time.perf_counter()
        s_bm_all = row_bm.score(qt)
        L_bm = top_idx(s_bm_all, K)
        TIM["BM25 (whole row)"].append(time.perf_counter() - _t)
        res["BM25 (whole row)"].append(metrics([int(r) for r in L_bm], gold))
        res["Dense (whole row)"].append(metrics([int(r) for r in L_dense], gold))
        _t = time.perf_counter()
        rrf2 = rrf([L_dense, L_bm], args.rrf_k)
        TIM["RRF fusion (after the two searches)"].append(time.perf_counter() - _t)
        rank_rrf2 = ranked_by(rrf2)
        res[REF].append(metrics(rank_rrf2, gold))

        _t = time.perf_counter()
        col_dense = {c: store.vecs[c] @ q for c in store.cols}
        col_bm = {c: store.bm25[c].score(qt) for c in store.cols}
        TIM["per-column scores, ALL rows (combo input)"].append(time.perf_counter() - _t)
        lists = [L_dense, L_bm]
        for c in store.cols:
            lists.append(store.rows[c][top_idx(col_dense[c], K)])
            lists.append(store.rows[c][top_idx(col_bm[c], K)])
        res["RRF  all columns (2C+2 lists)"].append(metrics(ranked_by(rrf(lists, args.rrf_k)), gold))
        if gl_all is not None:
            cand = [int(r) for r in L_bm]
            _t = time.perf_counter()
            bonus, nused = gliner_bonus(gl_all[qi], cand, corpus, gl_map)
            TIM["tag check, exact (100 rows, tags given)"].append(time.perf_counter() - _t)
            zb = zscore([s_bm_all[r] for r in cand])
            for lam, lab in GL_LINES:
                fin = zb + lam * bonus
                res[lab].append(metrics([cand[j] for j in np.argsort(-fin, kind="mergesort")], gold))
            gl_used.append(nused > 0)
            if args.gl_ceiling:
                M, used = gliner_match_matrix(gl_all[qi], cand, corpus, gl_map)
                sat = gl_satisfies(M, used)
                gset = set(gold)
                is_gold = np.array([r in gset for r in cand])
                rec = {"tagged": bool(used), "top1_ok": bool(is_gold[0]) if len(cand) else False,
                       "top1_violates": bool(used) and not bool(sat[0]),
                       "fixable": False, "sat_frac": float(sat.mean()) if len(sat) else 0.0}
                rec["fixable"] = rec["top1_violates"] and not rec["top1_ok"] and bool((sat & is_gold).any())
                gl_ceil.append(rec)
                if used:
                    _t = time.perf_counter()
                    Ms = gliner_match_stem(used, cand, corpus, gl_map)
                    Ss = gliner_match_sbert(used, cand, store, gl_map, doc_enc, args.use_names)
                    TIM["tag check, stem+sbert (100 rows, tags given)"].append(time.perf_counter() - _t)
                else:
                    Ms = np.zeros((len(cand), 0))
                    Ss = np.zeros((len(cand), 0))
                gl_mats.append({"qid": str(qrec.get("qid", "")), "qtype": qrec.get("qtype", ""),
                                "used": used, "E": M >= 1.0, "T": Ms >= 1.0, "S": Ss, "gold": is_gold,
                                "Ef": M, "Tf": Ms, "zb": np.asarray(zb, dtype=np.float64),
                                "ng": max(len(gold), 1),
                                "bm_m": res["BM25 (whole row)"][-1],
                                "margin": float((s_bm_all[cand[0]] - s_bm_all[cand[1]]) / s_bm_all[cand[0]])
                                if len(cand) > 1 and s_bm_all[cand[0]] > 0 else 0.0})
                # v33: standard label-free confidence signals (BM25 only), as extra selector baselines
                _sc = np.asarray(s_bm_all[list(cand)], dtype=np.float64)
                _s1 = _sc[0] if len(_sc) else 0.0
                _qt = set(tokenize(qrec["query"]))
                _top = " ".join(corpus[int(cand[0])]["fields"].get(c_, "") for c_ in columns) if len(cand) else ""
                gl_mats[-1]["qpp"] = {
                    "top5 vs rest": float((_sc[:5].mean() - _sc[19:].mean()) / _s1) if len(_sc) > 20 and _s1 > 0 else 0.0,
                    "NQC-like spread": float(_sc.std() / _sc.mean()) if len(_sc) and _sc.mean() > 0 else 0.0,
                    "WIG-like": float((_sc[:5].mean() - float(np.mean(s_bm_all))) / np.sqrt(max(len(_qt), 1))) if len(_sc) else 0.0,
                    "term coverage top1": float(len(_qt & set(tokenize(_top))) / len(_qt)) if _qt else 0.0,
                }
                if args.selective and ce:
                    P = min(args.ce_budget, len(cand))
                    docs = [row_text(corpus[r]["fields"], columns, args.use_names) for r in cand[:P]]
                    _t = time.perf_counter()
                    sce = np.asarray(ce.predict([(qrec["query"], d_) for d_ in docs],
                                                batch_size=args.ce_batch_size, show_progress_bar=False))
                    TIM[f"cross-encoder ({P} rows)"].append(time.perf_counter() - _t)
                    ranked = [cand[j] for j in np.argsort(-sce, kind="mergesort")] + cand[P:]
                    gl_mats[-1]["ce_m"] = metrics(ranked, gold)
                    gl_mats[-1]["sce"] = sce
                # two candidate policies, reported for design on the validation split only
                order_bm = np.arange(len(cand))
                filt = sorted(order_bm, key=lambda i: (not sat[i], i))           # hard filter
                viol = np.zeros(len(cand))
                for j, sp in enumerate(used):
                    viol += (M[:, j] if sp["neg"] else (1.0 - M[:, j])) * sp["score"]
                pen = np.argsort(zb - viol, kind="mergesort")[::-1]             # penalty only
                res[GL_FILTER].append(metrics([cand[i] for i in filt], gold))
                gated = list(order_bm) if sat[0] else filt                      # gate on top-1 violation
                res[GL_GATED].append(metrics([cand[i] for i in gated], gold))
                res[GL_PENALTY].append(metrics([cand[i] for i in pen], gold))
        if args.combo_new:
            cn_metrics = {}
            rels = {"names": {c: float(name_vecs[c] @ q) for c in store.cols},
                    "content": {c: float(centroids[c] @ q) for c in store.cols}}
            for label, srcname, mode in CN_VARIANTS:
                rel = rels[srcname]
                S_cn, cn_cols = combo_new_scores(rel, store, col_dense, len(corpus), args.cn_k, mode)
                if args.cn_pool == "bm25":
                    cand = np.asarray(L_bm, dtype=np.int64)  # re-order BM25's own top-K only
                    ranked_cn = [int(cand[j]) for j in np.argsort(-S_cn[cand], kind="mergesort")]
                else:
                    ranked_cn = [int(r) for r in top_idx(S_cn, K)]
                m = metrics(ranked_cn, gold)
                res[label].append(m)
                cn_metrics[label] = m
                if qi < 3 and mode == "zero":
                    print(f"  [{srcname}] q{qi + 1} columns chosen: "
                          + ", ".join(f"{c} ({rel[c]:.2f})" for c in cn_cols))

        # candidate pool: union of the two whole-row lists 
        pool = rank_rrf2
        ceiling.append(len(set(gold) & set(pool)) / len(gold))
        _t = time.perf_counter()
        cs, orders = combo_scores(pool, store, col_dense, col_bm, q, [0.0, bw], args)
        TIM["combo: combinations on the pool (needs column scores)"].append(time.perf_counter() - _t)
        base = zscore([rrf2[r] for r in pool])
        for label, w in (("combo dense   ", 0.0), ("combo hybrid  ", bw)):
            sc, _ = cs[w]
            for al in alphas:
                fused = al * base + (1 - al) * zscore(sc)
                res[f"{label} (alpha={al})"].append(metrics([pool[j] for j in np.argsort(-fused)], gold))
        # the frozen main configuration, used for fusion with the cross-encoder
        combo_main = args.freeze_alpha * base + (1 - args.freeze_alpha) * zscore(cs[bw][0])
        # POST-HOC variant: same mix, but the base is whole-row BM25 instead of RRF
        bm_base = zscore([s_bm_all[r] for r in pool])
        combo_bm = args.freeze_alpha * bm_base + (1 - args.freeze_alpha) * zscore(cs[bw][0])
        res[COMBO_BM25].append(metrics([pool[j] for j in np.argsort(-combo_bm)], gold))
        # ---- GATED: BM25 keeps the confident top, your combo orders the rest
        rank_combo = [pool[j] for j in np.argsort(-combo_main)]
        smax = float(s_bm_all[L_bm[0]]) if len(L_bm) else 0.0

        def gate_pos(tau):
            keep = [int(r) for r in L_bm if smax > 0 and s_bm_all[r] >= tau * smax]
            ks = set(keep)
            return keep + [r for r in rank_combo if r not in ks], len(keep)

        s2 = float(s_bm_all[L_bm[1]]) if len(L_bm) > 1 else 0.0
        margin = (smax - s2) / smax if smax > 0 else 0.0

        def gate_q(mu):
            use_bm = margin >= mu
            return ([int(r) for r in L_bm] if use_bm else rank_combo), use_bm

        if args.predictors:
            pred = confidence_predictors(s_bm_all, L_bm, L_dense, rank_combo, qt, corpus, columns)
            pred.update(failure_signals(row_bm, s_bm_all, L_bm, qt, corpus, columns))
            if args.combo_new:
                for lab in (CN_NAMES, CN_CONTENT):
                    pred_rows_cn[lab].append({"pred": pred, "bm": res["BM25 (whole row)"][-1],
                                              "combo": cn_metrics[lab]})
            pred_rows.append({"pred": pred, "bm": res["BM25 (whole row)"][-1],
                              "combo": metrics(rank_combo, gold)})
        rk, nk = gate_pos(args.gate_tau)
        res[GATE_POS].append(metrics(rk, gold))
        gate_kept.append(nk)
        rk, ub = gate_q(args.gate_margin)
        res[GATE_Q].append(metrics(rk, gold))
        gate_bm_used.append(float(ub))
        if args.gate_sweep:
            for t in SWEEP_TAU:
                res[f"  [sens] gate-pos tau={t}"].append(metrics(gate_pos(t)[0], gold))
            for m in SWEEP_MU:
                res[f"  [sens] gate-q margin={m}"].append(metrics(gate_q(m)[0], gold))

        if args.exact_ablation:
            sc_e, _ = combo_scores_exact(pool, store, col_dense, col_bm, q, bw, args, corpus, doc_enc)
            main_e = args.freeze_alpha * base + (1 - args.freeze_alpha) * zscore(sc_e)
            res[COMBO_EXACT].append(metrics([pool[j] for j in np.argsort(-main_e)], gold))

        best = dict(zip(pool, cs[bw][1]))
        order_map = dict(zip(pool, orders))
        if args.diagnose and len(diag) < args.diagnose:
            rank_main = [pool[j] for j in np.argsort(-combo_main)]
            gset = set(gold)
            if L_bm[0] in gset and rank_main[0] not in gset:
                pos_rank = {r: i + 1 for i, r in enumerate(rank_main)}
                mbm = max((float(col_bm[c].max()) for c in store.cols if len(col_bm[c])), default=0.0) or 1.0
                block = [f"### Q{len(diag) + 1}: {qrec['query']}", "",
                         "**Your method's top-1 (wrong):**"]
                block += explain_row(rank_main[0], corpus, store, col_dense, col_bm, mbm, bw, best, pos_rank)
                block += ["", "**Correct answer (BM25's top-1):**"]
                block += explain_row(int(L_bm[0]), corpus, store, col_dense, col_bm, mbm, bw, best, pos_rank)
                diag.append("\n".join(block) + "\n")
        if ce:
            P = min(args.ce_budget, len(rank_rrf2))
            top = rank_rrf2[:P]
            full = [row_text(corpus[r]["fields"], columns, args.use_names) for r in top]
            part = [row_text(corpus[r]["fields"], best.get(r, ()), args.use_names) or f
                    for r, f in zip(top, full)]
            ordered = [row_text(corpus[r]["fields"], order_map.get(r) or columns, args.use_names) or f
                       for r, f in zip(top, full)]
            ce_s = {}
            for key, docs in (("full", full), ("combo", part), ("ordered", ordered)):
                _t = time.perf_counter()
                ce_s[key] = np.asarray(ce.predict([(qrec["query"], d) for d in docs],
                                                  batch_size=args.ce_batch_size,
                                                  show_progress_bar=False), dtype=np.float64)
                if key == "full":
                    TIM[f"cross-encoder ({P} rows)"].append(time.perf_counter() - _t)
            pidx = {r: i for i, r in enumerate(pool)}
            cmb = zscore([combo_main[pidx[r]] for r in top])

            def rank_top(score):
                return [top[j] for j in np.argsort(-np.asarray(score))] + rank_rrf2[P:]

            m = args.ce_mix
            res[CE_FULL].append(metrics(rank_top(ce_s["full"]), gold))
            res[CE_COMBO].append(metrics(rank_top(ce_s["combo"]), gold))
            res[CE_ORDER].append(metrics(rank_top(ce_s["ordered"]), gold))
            res[CE_FULL_MIX].append(metrics(rank_top(m * zscore(ce_s["full"]) + (1 - m) * cmb), gold))
            res[CE_ORDER_MIX].append(metrics(rank_top(m * zscore(ce_s["ordered"]) + (1 - m) * cmb), gold))

        # ---- column filter: find rows whose two best columns match strongly (all rows)
        if args.col_filter:
            mbm = max((float(col_bm[c].max()) for c in store.cols if len(col_bm[c])), default=0.0) or 1.0
            b1 = np.full(len(corpus), -1e9)
            b2 = np.full(len(corpus), -1e9)
            for c in store.cols:
                rws = store.rows[c]
                sc_c = col_dense[c] + bw * col_bm[c] / mbm
                o1, o2 = b1[rws], b2[rws]
                up = sc_c > o1
                b2[rws] = np.where(up, o1, np.maximum(o2, sc_c))
                b1[rws] = np.where(up, sc_c, o1)
            proxy = b1 + np.where(b2 > -1e8, b2, 0.0)
            proxy[b1 <= -1e8] = -1e18
            L_col = top_idx(proxy, args.col_filter_k)
            rrf3 = rrf([L_dense, L_bm, L_col], args.rrf_k)
            rank_rrf3 = ranked_by(rrf3)
            res[RRF3].append(metrics(rank_rrf3, gold))
            pool2 = rank_rrf3  # = union of the three lists
            ceiling2.append(len(set(gold) & set(pool2)) / len(gold))
            cs2, orders2 = combo_scores(pool2, store, col_dense, col_bm, q, [bw], args)
            main2 = args.freeze_alpha * zscore([rrf3[r] for r in pool2]) \
                + (1 - args.freeze_alpha) * zscore(cs2[bw][0])
            rank_main2 = [pool2[j] for j in np.argsort(-main2)]
            res[COMBO_PLUS].append(metrics(rank_main2, gold))
            if ce:
                P2 = min(args.ce_budget, len(rank_main2))
                top2 = rank_main2[:P2]  # your method chooses which rows the reranker reads
                om2 = dict(zip(pool2, orders2))
                full2 = [row_text(corpus[r]["fields"], columns, args.use_names) for r in top2]
                ord2 = [row_text(corpus[r]["fields"], om2.get(r) or columns, args.use_names) or f
                        for r, f in zip(top2, full2)]
                s_full2 = np.asarray(ce.predict([(qrec["query"], d) for d in full2], batch_size=args.ce_batch_size,
                                                show_progress_bar=False), dtype=np.float64)
                s_ord2 = np.asarray(ce.predict([(qrec["query"], d) for d in ord2], batch_size=args.ce_batch_size,
                                               show_progress_bar=False), dtype=np.float64)
                pidx2 = {r: i for i, r in enumerate(pool2)}
                cmb2 = zscore([main2[pidx2[r]] for r in top2])

                def rank_top2(score):
                    return [top2[j] for j in np.argsort(-np.asarray(score))] + rank_main2[P2:]

                res[CE_PLUS_FULL].append(metrics(rank_top2(s_full2), gold))
                res[CE_PLUS_MIX].append(metrics(rank_top2(args.ce_mix * zscore(s_ord2)
                                                          + (1 - args.ce_mix) * cmb2), gold))

        runs.append({"qid": qrec["qid"], "query": qrec["query"],
                     "best_combos": {str(corpus[r]["id"]): list(best.get(r, ())) for r in pool[:5]},
                     "per_method": {k: v[-1] for k, v in res.items()}})
        if (qi + 1) % (1 if ce else 25) == 0:
            print(f"  {qi + 1}/{len(queries)} queries ({time.time() - t0:.0f}s)", flush=True)

    # ---- report
    print(f"\nResults over {len(queries)} queries (in %). Each list keeps its top {K}.")
    print(f"p-values: paired randomisation test against '{REF.strip()}' (p < 0.05 = likely real)\n")
    hdr = f"{'method':34s} {'Hit@1':>6s} {'Hit@5':>6s} {'R@20':>6s} {'MRR':>6s}   {'p(H@1)':>7s} {'p(MRR)':>7s}"
    print(hdr)
    print("-" * len(hdr))
    summary = {}
    for name, vals in res.items():
        if not vals:
            continue
        avg = {k: 100 * float(np.mean([v[k] for v in vals])) for k in vals[0]}
        if name == REF:
            p1 = pm = "  ref"
        else:
            p1 = f"{perm_test([v['Hit@1'] for v in vals], [v['Hit@1'] for v in res[REF]], args.perm):.3f}"
            pm = f"{perm_test([v['MRR'] for v in vals], [v['MRR'] for v in res[REF]], args.perm):.3f}"
        summary[name] = {**avg, "p_Hit@1": p1, "p_MRR": pm}
        print(f"{name:34s} " + " ".join(f"{avg[k]:6.1f}" for k in ("Hit@1", "Hit@5", "R@20", "MRR"))
              + f"   {p1:>7s} {pm:>7s}")
    main_combo = f"combo hybrid   (alpha={args.freeze_alpha})"
    BM = "BM25 (whole row)"
    pairs = [(main_combo, REF), (main_combo, BM), (COMBO_BM25, BM), (COMBO_BM25, main_combo)]
    if args.exact_ablation:
        pairs += [(COMBO_EXACT, main_combo), (COMBO_EXACT, BM)]
    pairs += [(GATE_POS, BM), (GATE_POS, main_combo), (GATE_Q, BM), (GATE_Q, main_combo)]
    if args.gliner:
        pairs += [(GL_MAIN, BM)]
    if args.combo_new:
        pairs += [(lab, BM) for lab, _, _ in CN_VARIANTS]
        pairs += [(CN_CONTENT, CN_NAMES), ("COMBO-NEW content avg-renorm", CN_CONTENT),
                  ("COMBO-NEW content decay", CN_CONTENT)]
    if ce:
        pairs += [(CE_FULL, REF), (CE_COMBO, CE_FULL), (CE_ORDER, CE_FULL),
                  (CE_FULL_MIX, CE_FULL), (CE_ORDER_MIX, CE_FULL)]
    if args.col_filter:
        pairs += [(RRF3, REF), (COMBO_PLUS, main_combo)]
        if ce:
            pairs += [(CE_PLUS_FULL, CE_FULL), (CE_PLUS_MIX, CE_FULL), (CE_PLUS_MIX, CE_ORDER_MIX)]
    print("\nDirect comparisons: difference in points (A - B) and paired p-value")
    print(f"{'A':34s} {'vs B':34s} {'dHit@1':>7s} {'p':>6s} {'dMRR':>7s} {'p':>6s}")
    direct = []
    for a, b in pairs:
        if not res.get(a) or not res.get(b):
            continue
        row = [a.strip(), b.strip()]
        line = f"{a:34s} {b:34s}"
        for k in ("Hit@1", "MRR"):
            va, vb = [v[k] for v in res[a]], [v[k] for v in res[b]]
            d = 100 * (np.mean(va) - np.mean(vb))
            pv = perm_test(va, vb, args.perm)
            row += [round(d, 2), round(pv, 4)]
            line += f" {d:+7.1f} {pv:6.3f}"
        direct.append(row)
        print(line)
    summary["_direct_comparisons"] = direct
    print(f"\ngate per-position: BM25 kept on average {np.mean(gate_kept):.1f} rows at the top"
          f" (tau={args.gate_tau}); gate per-question: BM25 used for {100 * np.mean(gate_bm_used):.0f}% of queries"
          f" (margin={args.gate_margin})")
    print(f"\nrecall ceiling, whole-row pool (BM25+dense):          {100 * np.mean(ceiling):.1f}%")
    if ceiling2:
        print(f"recall ceiling, POOL+ (with your column filter):      {100 * np.mean(ceiling2):.1f}%")
    sfx = f"_{args.tag}" if args.tag else ""
    if args.timing:
        nt = min(args.timing_n, len(queries))
        texts = [qq["query"] for qq in queries[:nt]]
        q_enc.encode(texts[:2])  # warm-up
        for t_ in texts:
            _t = time.perf_counter()
            q_enc.encode([t_])
            TIM["encode the question (SBERT)"].append(time.perf_counter() - _t)
        if args.gliner:
            try:
                from gliner import GLiNER
                gm = GLiNER.from_pretrained(args.gl_model)
                try:
                    gm = gm.to(args.device or "cpu")
                except Exception:  # noqa: BLE001
                    pass
                gm.predict_entities(texts[0], list(gl_map), threshold=args.gl_threshold)  # warm-up
                for t_ in texts:
                    _t = time.perf_counter()
                    gm.predict_entities(t_, list(gl_map), threshold=args.gl_threshold)
                    TIM["GLiNER tagging of the question"].append(time.perf_counter() - _t)
            except Exception as e:  # noqa: BLE001
                print(f"(GLiNER timing skipped: {e})")
        print(f"\n=== TIMING per question (device: {args.device or 'cpu'}; median / mean in milliseconds) ===")
        print(f"{'step':52s} {'median':>9s} {'mean':>9s} {'n':>5s}")
        tout = {}
        for k_, v_ in TIM.items():
            v_ = np.asarray(v_[1:] if len(v_) > 3 else v_) * 1000  # drop the first (warm-up) call
            tout[k_] = {"median_ms": float(np.median(v_)), "mean_ms": float(np.mean(v_)), "n": int(len(v_))}
            print(f"{k_:52s} {np.median(v_):9.2f} {np.mean(v_):9.2f} {len(v_):5d}")
        print("(one-time costs such as embedding the table or building indexes are NOT included)")
        with open(os.path.join(args.data, f"timing{sfx}.json"), "w") as f:
            json.dump(tout, f, indent=2)
    if args.gliner:
        g = np.array(gl_used, bool)
        bm = res["BM25 (whole row)"]
        gm = res[GL_MAIN]
        print(f"\n=== GLiNER+BM25: questions with tags = {int(g.sum())}/{len(g)} "
              f"(untagged questions are scored exactly as BM25) ===")
        for k in ("Hit@1", "MRR"):
            a = np.array([m[k] for m in gm])[g]
            b = np.array([m[k] for m in bm])[g]
            if len(a):
                print(f"  on tagged questions only: {k} BM25 {100 * b.mean():.1f} -> {100 * a.mean():.1f} "
                      f"(p={perm_test(a, b, args.perm):.3f}); "
                      f"better on {int((a > b).sum())}, worse on {int((a < b).sum())}")
    if args.gliner and args.gl_ceiling and gl_ceil:
        n = len(gl_ceil)
        T = sum(r["tagged"] for r in gl_ceil)
        V = sum(r["top1_violates"] for r in gl_ceil)
        VW = sum(r["top1_violates"] and not r["top1_ok"] for r in gl_ceil)
        VC = sum(r["top1_violates"] and r["top1_ok"] for r in gl_ceil)
        F = sum(r["fixable"] for r in gl_ceil)
        bm1 = 100 * np.mean([m["Hit@1"] for m in res["BM25 (whole row)"]])
        print("\n=== GLiNER ceiling: how much could ANY tag-based policy fix? ===")
        print(f"questions with tags:                                   {T}/{n}")
        print(f"BM25's top-1 VIOLATES a tag:                           {V}/{n}")
        print(f"  ...and is wrong (candidates to fix):                 {VW}")
        print(f"  ...but is actually correct (a policy could break it): {VC}")
        print(f"FIXABLE (top-1 wrong & violates, and a correct row in the top {K} satisfies all tags): {F}")
        print(f"CEILING of any tag policy: Hit@1 {bm1:.1f} -> {bm1 + 100 * F / n:.1f}  (+{100 * F / n:.1f})")
        print(f"average share of BM25's top {K} that satisfies all tags: "
              f"{100 * np.mean([r['sat_frac'] for r in gl_ceil if r['tagged']] or [0]):.0f}%")
        print("rule of thumb: ceiling under ~+5 -> no tag policy can win; 10+ -> worth designing (on validation)")
        print("\n--- Matching sweep: how should 'the tag is present in its column' be decided? ---")
        print("modes: exact words | stem (shoe=shoes) | sbert>=tau | stem OR sbert>=tau")
        print(f"{'mode':22s} {'fires':>6s} {'right':>6s} {'wrong':>6s} {'FIXABLE':>8s} {'ceiling H@1':>12s} {'% top100 pass':>14s}")
        configs = [("exact", None), ("stem", None)]
        configs += [("sbert", t) for t in args.gl_taus]
        configs += [("stem|sbert", t) for t in args.gl_taus]
        sweep = {}
        for mode, tau in configs:
            fires = right = wrong = fix = 0
            passfrac = []
            for d in gl_mats:
                if not d["used"]:
                    continue
                if mode == "exact":
                    P = d["E"]
                elif mode == "stem":
                    P = d["T"]
                elif mode == "sbert":
                    P = d["S"] >= tau
                else:
                    P = d["T"] | (d["S"] >= tau)
                sat = gl_satisfies_soft(P, d["used"])
                passfrac.append(sat.mean())
                if not sat[0]:
                    fires += 1
                    if d["gold"][0]:
                        wrong += 1
                    else:
                        right += 1
                        if (sat & d["gold"]).any():
                            fix += 1
            name = mode if tau is None else f"{mode} {tau}"
            sweep[name] = {"fires": fires, "right": right, "wrong": wrong, "fixable": fix,
                           "ceiling": bm1 + 100 * fix / n, "pass_pct": 100 * float(np.mean(passfrac or [0]))}
            print(f"{name:22s} {fires:6d} {right:6d} {wrong:6d} {fix:8d} {bm1 + 100 * fix / n:12.1f} "
                  f"{100 * float(np.mean(passfrac or [0])):13.0f}%")
        print("(fires = BM25 top-1 fails the check; right = it was wrong; wrong = it was correct)")

        print("\n--- RELATIVE version: score = share of tags a row satisfies; the gate fires when some row in the"
              f" top {K} covers MORE tags than BM25's top-1 ---")
        print(f"{'mode':22s} {'fires':>6s} {'right':>6s} {'wrong':>6s} {'FIXABLE':>8s} {'ceiling H@1':>12s} {'H@1 if re-sorted':>17s}")
        for mode, tau in configs:
            fires = right = wrong = fix = 0
            hits = []
            for d in gl_mats:
                if not d["used"]:
                    hits.append(float(d["gold"][0]))
                    continue
                if mode == "exact":
                    P = d["E"]
                elif mode == "stem":
                    P = d["T"]
                elif mode == "sbert":
                    P = d["S"] >= tau
                else:
                    P = d["T"] | (d["S"] >= tau)
                sc = np.zeros(P.shape[0])
                for j, sp in enumerate(d["used"]):
                    sc += (~P[:, j]) if sp["neg"] else P[:, j]
                sc = sc / len(d["used"])
                order = sorted(range(len(sc)), key=lambda i: (-sc[i], i))  # more tags first, then BM25 order
                hits.append(float(d["gold"][order[0]]))
                if sc[0] < sc.max():
                    fires += 1
                    if d["gold"][0]:
                        wrong += 1
                    else:
                        right += 1
                        if (d["gold"] & (sc > sc[0])).any():
                            fix += 1
            name = mode if tau is None else f"{mode} {tau}"
            print(f"{name:22s} {fires:6d} {right:6d} {wrong:6d} {fix:8d} {bm1 + 100 * fix / n:12.1f} "
                  f"{100 * float(np.mean(hits)):16.1f}")
        print(f"(H@1 if re-sorted = actually putting the rows with most tags first, BM25 order within ties;"
              f" compare with BM25's {bm1:.1f})")

        print("\n--- PARTIAL CREDIT: credit per tag = share of the tag's words found (or SBERT similarity);"
              " product score = average credit (negated tags: 1 - credit) ---")
        print(f"{'mode':26s} {'fires':>6s} {'right':>6s} {'wrong':>6s} {'FIXABLE':>8s} {'H@1 if re-sorted':>17s} {'MRR':>6s}")
        pmodes = [("exact words, partial", "Ef"), ("stem words, partial", "Tf"),
                  ("sbert similarity", "S"), ("max(stem partial, sbert)", "TfS")]
        partial = {}
        for name, key in pmodes:
            fires = right = wrong = fix = 0
            hits, rr = [], []
            for d in gl_mats:
                g = d["gold"]
                if not d["used"]:
                    hits.append(float(g[0]))
                    first = int(np.argmax(g)) if g.any() else None
                    rr.append(0.0 if first is None else 1.0 / (first + 1))
                    continue
                if key == "TfS":
                    C = np.maximum(d["Tf"], np.clip(d["S"], 0, 1))
                elif key == "S":
                    C = np.clip(d["S"], 0, 1)
                else:
                    C = d[key]
                sc = np.zeros(C.shape[0])
                for j, sp in enumerate(d["used"]):
                    sc += (1.0 - C[:, j]) if sp["neg"] else C[:, j]
                sc = np.round(sc / len(d["used"]), 6)
                order = sorted(range(len(sc)), key=lambda i: (-sc[i], i))
                hits.append(float(g[order[0]]))
                pos = [k for k, i in enumerate(order) if g[i]]
                rr.append(1.0 / (pos[0] + 1) if pos else 0.0)
                if sc[0] < sc.max():
                    fires += 1
                    if g[0]:
                        wrong += 1
                    else:
                        right += 1
                        if (g & (sc > sc[0])).any():
                            fix += 1
            partial[name] = {"fires": fires, "right": right, "wrong": wrong, "fixable": fix,
                             "hit1": 100 * float(np.mean(hits)), "mrr": 100 * float(np.mean(rr))}
            print(f"{name:26s} {fires:6d} {right:6d} {wrong:6d} {fix:8d} {100 * np.mean(hits):16.1f} "
                  f"{100 * np.mean(rr):6.1f}")
        bm_mrr = 100 * np.mean([m["MRR"] for m in res["BM25 (whole row)"]])
        print(f"(BM25 alone: H@1 {bm1:.1f}, MRR {bm_mrr:.1f}; MRR here is within BM25's top {K})")

        # ---------------- design grid: better replacement / fewer overrules / soft mix
        def tag_count(d, mode):
            if mode == "stem":
                P = d["T"]
            elif mode.startswith("stem|sbert"):
                P = d["T"] | (d["S"] >= float(mode.split()[-1]))
            elif mode == "stem partial":
                P = d["Tf"]
            else:
                P = d["E"]
            P = P.astype(float)
            sc = np.zeros(P.shape[0])
            for j, sp in enumerate(d["used"]):
                sc += (1.0 - P[:, j]) if sp["neg"] else P[:, j]
            return sc  # in units of tags

        def evaluate(policy):
            hits, rr, h5, r20 = [], [], [], []
            fires = right = wrong = repl_ok = 0
            for d in gl_mats:
                g = d["gold"]
                order = policy(d) if d["used"] else list(range(len(g)))
                if order[0] != 0:
                    fires += 1
                    if g[0]:
                        wrong += 1
                    else:
                        right += 1
                        repl_ok += int(g[order[0]])
                hits.append(float(g[order[0]]))
                pos = [k for k, i in enumerate(order) if g[i]]
                rr.append(1.0 / (pos[0] + 1) if pos else 0.0)
                h5.append(float(bool(pos) and pos[0] < 5))
                r20.append(sum(1 for k in pos if k < 20) / d["ng"])
            evaluate.extra = (100 * np.mean(h5), 100 * np.mean(r20))
            return 100 * np.mean(hits), 100 * np.mean(rr), fires, right, wrong, repl_ok

        def promote(kmax, need):
            def pol(d, mode):
                sc = tag_count(d, mode)
                top = min(kmax, len(sc))
                best = max(range(top), key=lambda i: (sc[i], -i))
                if sc[best] - sc[0] >= need - 1e-9:
                    return [best] + [i for i in range(len(sc)) if i != best]
                return list(range(len(sc)))
            return pol

        def soft(lam):
            def pol(d, mode):
                fin = d["zb"] + lam * tag_count(d, mode)
                return list(np.argsort(-fin, kind="mergesort"))
            return pol

        policies = [(f"promote top-{k}, +{n} tag", promote(k, n)) for k in (3, 5, 10) for n in (1, 2)]
        policies += [(f"soft mix lambda={l}", soft(l)) for l in (0.5, 1.0, 2.0, 4.0)]
        dmodes = ["stem", "stem partial", "stem|sbert 0.7", "stem|sbert 0.8"]
        print("\n=== DESIGN GRID (validation only): pick ONE, freeze it, then test once ===")
        print("promote top-k, +n tag: move to #1 the row in BM25's top k with the most tags, only if it has")
        print("  at least n more tags than BM25's #1 | soft mix: BM25 (standardized) + lambda x number of tags")
        print(f"replacement must be correct > wrong/right of the time to beat BM25 ({bm1:.1f} H@1, {bm_mrr:.1f} MRR)\n")
        bm5 = 100 * np.mean([m["Hit@5"] for m in res["BM25 (whole row)"]])
        bm20 = 100 * np.mean([m["R@20"] for m in res["BM25 (whole row)"]])
        print(f"BM25 alone: H@1 {bm1:.1f}  H@5 {bm5:.1f}  R@20 {bm20:.1f}  MRR {bm_mrr:.1f}\n")
        print(f"{'mode':16s} {'policy':24s} {'H@1':>6s} {'H@5':>6s} {'R@20':>6s} {'MRR':>6s} {'fires':>6s} {'right':>6s} {'wrong':>6s} {'repl ok':>8s}")
        grid = {}
        for mode in dmodes:
            for pname, pol in policies:
                h1, mrr, fi, ri, wr, ok = evaluate(lambda d, pol=pol, mode=mode: pol(d, mode))
                hh5, rr20 = evaluate.extra
                grid[f"{mode} | {pname}"] = {"H@1": h1, "H@5": hh5, "R@20": rr20, "MRR": mrr, "fires": fi,
                                              "right": ri, "wrong": wr, "replacement_correct": ok}
                flag = "  <-- beats BM25" if (h1 > bm1 and mrr > bm_mrr) else ""
                print(f"{mode:16s} {pname:24s} {h1:6.1f} {hh5:6.1f} {rr20:6.1f} {mrr:6.1f} {fi:6d} {ri:6d} {wr:6d} {ok:8d}{flag}")
            print()
        with open(os.path.join(args.data, f"gl_design{'_' + args.tag if args.tag else ''}.json"), "w") as f:
            json.dump(grid, f, indent=2)
        print("(repl ok = among 'right' fires, how often the new #1 is correct; a policy gains repl ok and loses wrong)")

        if args.selective and gl_mats and "ce_m" in gl_mats[0]:
            keys = ("Hit@1", "Hit@5", "R@20", "MRR")
            nq = len(gl_mats)
            bmM = {k: np.array([d["bm_m"][k] for d in gl_mats]) for k in keys}
            ceM = {k: np.array([d["ce_m"][k] for d in gl_mats]) for k in keys}
            mode = args.sel_mode
            alarm = np.array([(tag_count(d, mode).max() - tag_count(d, mode)[0]) if d["used"] else 0.0
                              for d in gl_mats])
            margin = np.array([d["margin"] for d in gl_mats])
            fired = alarm > 0
            # v32: per-question dump so significance can be tested offline (sel_stats.py)
            save_jsonl(os.path.join(args.data, f"selective_pq{'_' + args.tag if args.tag else ''}.jsonl"),
                       [{"qid": i, "qid_str": d.get("qid", ""), "qtype": d.get("qtype", ""), "bm": {k: float(d["bm_m"][k]) for k in keys},
                         "ce": {k: float(d["ce_m"][k]) for k in keys},
                         "alarm": float(alarm[i]), "margin": float(margin[i]),
                         "probe3_agree": bool(int(np.argmax(d["sce"][:3])) == 0),
                         "qpp": d.get("qpp", {})}
                        for i, d in enumerate(gl_mats)])
            # selection orders (first = most worth reranking)
            orders = {
                f"tag alarm ({mode})": np.lexsort((margin, -alarm)),   # strongest alarm first, ties: small margin
                "small BM25 margin": np.argsort(margin, kind="mergesort"),
            }
            gain = ceM["MRR"] - bmM["MRR"]
            orders["ORACLE (uses labels)"] = np.argsort(-gain, kind="mergesort")
            rng = np.random.default_rng(args.seed)

            def result(sel):
                return {k: 100 * float(np.where(sel, ceM[k], bmM[k]).mean()) for k in keys}

            print(f"\n=== SELECTIVE RERANKING: cross-encoder only on the questions a selector picks ===")
            print(f"cross-encoder reads BM25's top {args.ce_budget}; other questions keep BM25's ranking")
            r0, r1 = result(np.zeros(nq, bool)), result(np.ones(nq, bool))
            print(f"BM25 only (0% reranked):      H@1 {r0['Hit@1']:5.1f}  MRR {r0['MRR']:5.1f}")
            print(f"reranker on ALL (100%):       H@1 {r1['Hit@1']:5.1f}  MRR {r1['MRR']:5.1f}")
            fr = result(fired)
            rnd = [result(rng.permutation(nq) < int(fired.sum())) for _ in range(200)]
            print(f"tag alarm fires on {int(fired.sum())}/{nq} ({100 * fired.mean():.0f}%): "
                  f"H@1 {fr['Hit@1']:5.1f}  MRR {fr['MRR']:5.1f}   | random same budget: "
                  f"H@1 {np.mean([x['Hit@1'] for x in rnd]):5.1f}  MRR {np.mean([x['MRR'] for x in rnd]):5.1f}")
            budgets = (10, 20, 30, 40, 50, 60, 80)
            print(f"\n{'selector':26s} " + " ".join(f"{str(b) + '%':>13s}" for b in budgets))
            print(f"{'':26s} " + " ".join(f"{'H@1 / MRR':>13s}" for _ in budgets))
            sel_out = {}
            for name, order in list(orders.items()) + [("random (avg of 200)", None)]:
                cells, rec = [], {}
                for b in budgets:
                    k = int(round(nq * b / 100))
                    if order is None:
                        rs = [result(rng.permutation(nq) < k) for _ in range(200)]
                        r = {kk: float(np.mean([x[kk] for x in rs])) for kk in keys}
                    else:
                        sel = np.zeros(nq, bool)
                        sel[order[:k]] = True
                        r = result(sel)
                    rec[b] = r
                    cells.append(f"{r['Hit@1']:5.1f} / {r['MRR']:5.1f}")
                sel_out[name] = rec
                print(f"{name:26s} " + " ".join(f"{c:>13s}" for c in cells))
            print("(share of the reranker's gain kept = (selector - BM25) / (all - BM25))")
            # ---------------- PROBE: let the reranker look at BM25's top-k first
            def order_metrics(g, order, ngold):
                pos = [k_ for k_, i in enumerate(order) if g[i]]
                return {"Hit@1": float(bool(pos) and pos[0] == 0), "Hit@5": float(bool(pos) and pos[0] < 5),
                        "R@20": sum(1 for k_ in pos if k_ < 20) / ngold,
                        "MRR": 1.0 / (pos[0] + 1) if pos else 0.0}

            Pfull = len(gl_mats[0]["sce"])
            print(f"\n=== PROBE: the reranker first scores BM25's top k; if its #1 is BM25's #1 (agree),"
                  f" stop; otherwise rerank all {Pfull} ===")
            print("cost = reranker calls as % of reranking all questions fully; selectors below get the SAME cost")
            print(f"{'probe':30s} {'flagged':>8s} {'cost':>6s} {'H@1':>6s} {'MRR':>6s}  |"
                  f" {'margin H@1/MRR':>15s} {'random H@1/MRR':>15s} {'oracle H@1/MRR':>15s}")
            probe_out = {}
            for kprobe in (3, 5, 10):
                for variant in ("keep BM25 if agree", "reorder top-k if agree"):
                    ms, flag = [], []
                    for d in gl_mats:
                        g, sc_ = d["gold"], d["sce"]
                        agree = int(np.argmax(sc_[:kprobe])) == 0
                        flag.append(not agree)
                        if not agree:
                            ms.append(d["ce_m"])
                        elif variant.startswith("keep"):
                            ms.append(d["bm_m"])
                        else:
                            o = list(np.argsort(-sc_[:kprobe], kind="mergesort")) + list(range(kprobe, len(g)))
                            ms.append(order_metrics(g, o, d["ng"]))
                    flag = np.array(flag)
                    cost = (kprobe + (Pfull - kprobe) * flag.mean()) / Pfull
                    r = {k: 100 * float(np.mean([m_[k] for m_ in ms])) for k in keys}
                    ksel = int(round(nq * cost))
                    def at(order):
                        sel = np.zeros(nq, bool)
                        sel[order[:ksel]] = True
                        return result(sel)
                    rm = at(orders["small BM25 margin"])
                    ro = at(orders["ORACLE (uses labels)"])
                    rr = [result(rng.permutation(nq) < ksel) for _ in range(200)]
                    rr = {k: float(np.mean([x[k] for x in rr])) for k in keys}
                    name = f"top-{kprobe}, {variant}"
                    probe_out[name] = {"flagged_pct": 100 * float(flag.mean()), "cost_pct": 100 * cost, **r,
                                       "margin_same_cost": rm, "random_same_cost": rr, "oracle_same_cost": ro}
                    beat = "  <-- beats margin" if (r["Hit@1"] > rm["Hit@1"] and r["MRR"] > rm["MRR"]) else ""
                    print(f"{name:30s} {100 * flag.mean():7.0f}% {100 * cost:5.0f}% {r['Hit@1']:6.1f} {r['MRR']:6.1f}  |"
                          f" {rm['Hit@1']:6.1f} / {rm['MRR']:5.1f}  {rr['Hit@1']:6.1f} / {rr['MRR']:5.1f}"
                          f"  {ro['Hit@1']:6.1f} / {ro['MRR']:5.1f}{beat}")
            print(f"(reference: BM25 only H@1 {r0['Hit@1']:.1f} MRR {r0['MRR']:.1f}; rerank all H@1 {r1['Hit@1']:.1f} MRR {r1['MRR']:.1f})")
            with open(os.path.join(args.data, f"probe{'_' + args.tag if args.tag else ''}.json"), "w") as f:
                json.dump(probe_out, f, indent=2)
            with open(os.path.join(args.data, f"selective{'_' + args.tag if args.tag else ''}.json"), "w") as f:
                json.dump({"bm25": r0, "all": r1, "fired": fr, "fired_n": int(fired.sum()), "budgets": sel_out}, f, indent=2)
        with open(os.path.join(args.data, f"gl_partial{'_' + args.tag if args.tag else ''}.json"), "w") as f:
            json.dump(partial, f, indent=2)
        with open(os.path.join(args.data, f"gl_sweep{'_' + args.tag if args.tag else ''}.json"), "w") as f:
            json.dump(sweep, f, indent=2)
        if V:
            print(f"GATE quality (fires when top-1 violates a tag): fires on {V}; right to fire on {VW} "
                  f"({100 * VW / V:.0f}%), wrong on {VC} ({100 * VC / V:.0f}%)")
    if args.combo_new:
        bm5 = np.array([m["Hit@5"] for m in res["BM25 (whole row)"]])
        bm1 = np.array([m["Hit@1"] for m in res["BM25 (whole row)"]])
        print(f"\n=== COMBO-NEW as a fallback: rescue ceiling (candidates: {args.cn_pool}) ===")
        for lab, _, _ in CN_VARIANTS:
            c1 = np.array([m["Hit@1"] for m in res[lab]])
            bad = bm5 == 0
            resc = bad & (c1 == 1)
            print(f"{lab.strip():34s} BM25 fails (nothing in top 5) on {int(bad.sum())}/{len(bm5)}; "
                  f"rescued at #1: {int(resc.sum())};  ceiling Hit@1 {100 * bm1.mean():.1f} -> "
                  f"{100 * np.where(resc, 1, bm1).mean():.1f}")
        print("(for comparison, the old combo's ceiling is in the asymmetric section with --predictors)")
    if args.predictors and pred_rows and args.combo_new:
        for lab in (CN_NAMES, CN_CONTENT):
            report_asymmetric(pred_rows_cn[lab], args.perm, title=lab.strip())
    if args.predictors and pred_rows:
        pr = report_predictors(pred_rows, args.perm)
        pr["_asymmetric"] = report_asymmetric(pred_rows, args.perm)
        with open(os.path.join(args.data, f"predictors{sfx}.json"), "w") as f:
            json.dump(pr, f, indent=2)
        save_jsonl(os.path.join(args.data, f"predictors_per_query{sfx}.jsonl"), pred_rows)
    if args.diagnose:
        dfile = os.path.join(args.data, f"diagnose{sfx}.md")
        with open(dfile, "w") as f:
            f.write("# Cases where BM25 is right at rank 1 and your method is wrong\n\n"
                    "score = cosine + w * BM25 / max BM25 (per column); top 6 columns shown\n\n")
            f.write("\n".join(diag))
        print(f"\nwrote {len(diag)} diagnostic cases to {dfile}")
    out = os.path.join(args.data, f"results{sfx}.json")
    with open(out, "w") as f:
        json.dump({"args": vars(args), "summary": summary,
                   "pool_ceiling": 100 * float(np.mean(ceiling)),
                   "pool_plus_ceiling": 100 * float(np.mean(ceiling2)) if ceiling2 else None}, f, indent=2)
    save_jsonl(os.path.join(args.data, f"per_query{sfx}.jsonl"), runs)
    print(f"saved {out} and per_query{sfx}.jsonl  ({time.time() - t0:.0f}s total)")


# ------------------------------------------------------------------- toy data

def cmd_make_toy(args):
    os.makedirs(args.out, exist_ok=True)
    rnd = random.Random(0)
    brands = ["Adidas", "Nike", "Puma", "Reebok", "Asics"]
    kinds = ["running shoes", "sandals", "hiking boots", "tennis shoes", "slippers"]
    feels = ["very comfortable and stylish", "stiff and uncomfortable",
             "cheap looking", "great grip on trails", "breathable and light"]
    corpus, queries = [], []
    for i in range(400):
        b, k, f = rnd.choice(brands), rnd.choice(kinds), rnd.choice(feels)
        other = rnd.choice([x for x in brands if x != b])
        corpus.append({"id": str(i), "fields": {
            "brand": b, "title": f"{b} {k}",
            "description": f"{k} made of {rnd.choice(['mesh', 'leather', 'nubuck', 'canvas'])}",
            "review": f"{f}. Much better than my old {other} pair. " + "filler text " * rnd.randint(5, 60)}})
    for j in range(60):
        b, k, f = rnd.choice(brands), rnd.choice(kinds), rnd.choice(feels)
        gold = [r["id"] for r in corpus if r["fields"]["brand"] == b
                and k in r["fields"]["title"] and r["fields"]["review"].startswith(f)]
        if gold:
            queries.append({"qid": f"q{j}", "query": f"{b} {k} that are {f}", "answers": gold})
    save_jsonl(os.path.join(args.out, "corpus.jsonl"), corpus)
    save_jsonl(os.path.join(args.out, "queries.jsonl"), queries)
    print(f"wrote {len(corpus)} rows and {len(queries)} queries to {args.out}/")


# --------------------------------------------------------------------- STaRK

def load_stark_questions(name, split):
    """Read STaRK questions straight from the HuggingFace dataset repo.
    (Bypasses stark_qa.load_qa, whose download helper crashes in some versions.)"""
    import ast
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download
    repo = "snap-stanford/stark"
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    csvs = [f for f in files if f.startswith(f"qa/{name}/") and f.endswith("/stark_qa.csv")]
    idxf = [f for f in files if f.startswith(f"qa/{name}/") and f.endswith(f"/{split}.index")]
    if not csvs or not idxf:
        raise RuntimeError(f"Could not find STaRK question files for '{name}' / split '{split}'.")
    df = pd.read_csv(hf_hub_download(repo, csvs[0], repo_type="dataset")).set_index("id")
    with open(hf_hub_download(repo, idxf[0], repo_type="dataset")) as f:
        qids = [int(x) for x in f.read().split()]  # the .index file stores query ids
    out = []
    for qid in qids:
        if qid not in df.index:
            continue
        row = df.loc[qid]
        ans = row["answer_ids"]
        ans = ast.literal_eval(ans) if isinstance(ans, str) else list(ans)
        out.append((str(row["query"]), qid, [int(a) for a in ans]))
    return out


def stark_node_name(info):
    for key in ("name", "title", "brand_name", "category_name", "color_name", "DisplayName"):
        if info.get(key):
            return clean(info[key], 200)
    for v in info.values():  # first short text value as a fallback
        if isinstance(v, str) and 0 < len(v) < 200:
            return clean(v)
    return ""


def cmd_prepare_stark(args):
    """Best-effort converter. Text attributes become columns; nested dicts are split
    into one column per key; each relation type becomes a column listing the names
    of linked items (e.g. brand, parent/child, associated-with)."""
    from stark_qa import load_skb
    skb = load_skb(args.name, download_processed=True)

    questions = load_stark_questions(args.name, args.split)
    random.Random(args.seed).shuffle(questions)
    if args.max_queries:
        questions = questions[: args.max_queries]
    queries, gold_ids = [], set()
    for query, q_id, answer_ids in questions:
        queries.append({"qid": str(q_id), "query": query, "answers": [str(a) for a in answer_ids]})
        gold_ids.update(answer_ids)

    cand = list(getattr(skb, "candidate_ids", range(skb.num_nodes())))
    if args.sample_rows and args.sample_rows < len(cand):
        others = [c for c in cand if c not in gold_ids]
        random.Random(args.seed).shuffle(others)
        cand = sorted(gold_ids | set(others[: max(0, args.sample_rows - len(gold_ids))]))

    rels = []
    if args.relations:
        try:
            rels = list(skb.rel_type_lst())
            print(f"relation types used as columns: {rels}")
        except Exception as e:  # noqa: BLE001
            print(f"warning: could not read relation types ({e}); skipping relation columns")

    # Fast path: read every edge once and build (node, relation) -> neighbours.
    nbr_map, ei, et, etd = None, None, None, None
    if rels:
        try:
            t1 = time.time()
            ei, et, etd = skb.edge_index, skb.edge_types, skb.edge_type_dict
            ei = ei.numpy() if hasattr(ei, "numpy") else np.asarray(ei)
            et = et.numpy() if hasattr(et, "numpy") else np.asarray(et)
            mask = np.isin(ei[0], np.asarray(cand))
            nbr_map = {}
            for s, d, t in zip(ei[0][mask].tolist(), ei[1][mask].tolist(), et[mask].tolist()):
                lst = nbr_map.setdefault((s, etd[t]), [])
                if not args.max_neighbors or len(lst) < args.max_neighbors:
                    lst.append(d)
            print(f"indexed {int(mask.sum())} links in {time.time() - t1:.0f}s")
        except Exception as e:  # noqa: BLE001
            print(f"note: fast link index unavailable ({e}); using slow per-node lookups")
            nbr_map = None

    # MAG: like mFAR, write each author as "Name (Institution, ...)" by following
    # paper -> author -> institution links.
    aff_map = None
    AFF, WRITES = "author___affiliated_with___institution", "author___writes___paper"
    if nbr_map is not None and AFF in rels and WRITES in rels:
        inv = {v: k for k, v in etd.items()}
        m = et == inv[AFF]
        aff_map = {}
        for a, i in zip(ei[0][m].tolist(), ei[1][m].tolist()):
            aff_map.setdefault(a, []).append(i)
        print(f"author-institution links: {int(m.sum())}")

    wanted = args.fields.split(",") if args.fields else None
    skip = {"id", "node_id", "type", "node_type", "imUrl", "imgUrl", "asin"}
    name_cache = {}

    def node_name(n):
        n = int(n)
        if n not in name_cache:
            name_cache[n] = stark_node_name(skb.node_info[n])
        return name_cache[n]
    corpus = []
    for ci, c in enumerate(cand):
        if (ci + 1) % 10000 == 0:
            print(f"  built {ci + 1}/{len(cand)} rows")
        info = skb.node_info[c] if hasattr(skb, "node_info") else vars(skb[c])
        fields = {}
        for k, v in info.items():
            if not wanted and k in skip:
                continue
            items = v.items() if (isinstance(v, dict) and args.flatten_dicts) else [(None, v)]
            for sub, val in items:
                key = f"{k}.{sub}" if sub is not None else k
                if wanted and key not in wanted and k not in wanted:
                    continue
                t = clean(val, args.max_chars)
                if t:
                    fields[key] = t
        for rel in rels:
            if wanted and rel not in wanted:
                continue
            if nbr_map is not None:
                nbrs = nbr_map.get((c, rel), [])
            else:
                try:
                    nbrs = list(skb.get_neighbor_nodes(c, rel))
                    if args.max_neighbors:
                        nbrs = nbrs[: args.max_neighbors]
                except Exception:  # noqa: BLE001
                    continue
            names = []
            for n in nbrs:
                n = int(n)
                if n not in name_cache:
                    name_cache[n] = stark_node_name(skb.node_info[n])
                if name_cache[n]:
                    names.append(name_cache[n])
            if names:
                fields[rel] = clean(" ; ".join(names), args.max_chars)
        if aff_map is not None and (not wanted or AFF in wanted):
            parts = []
            for a in nbr_map.get((c, WRITES), []):
                nm = node_name(a)
                if not nm:
                    continue
                insts = [x for x in (node_name(i) for i in aff_map.get(int(a), [])) if x]
                parts.append(f"{nm} ({', '.join(insts)})" if insts else nm)
            if parts:
                fields[AFF] = clean(", ".join(parts), args.max_chars)
            else:
                fields.pop(AFF, None)
        corpus.append({"id": str(c), "fields": fields})

    os.makedirs(args.out, exist_ok=True)
    save_jsonl(os.path.join(args.out, "corpus.jsonl"), corpus)
    save_jsonl(os.path.join(args.out, "queries.jsonl"), queries)
    cols = sorted({k for r in corpus for k in r["fields"]})
    print(f"wrote {len(corpus)} rows, {len(queries)} queries; columns found: {cols}")


# ------------------------------------------------------ questions only (new split)

def cmd_prepare_queries(args):
    """Write queries.jsonl for another STaRK split WITHOUT loading the knowledge base, and link the
    table and all saved embeddings from an existing prepared folder (--like)."""
    src = os.path.abspath(args.like)
    if not os.path.exists(os.path.join(src, "corpus.jsonl")):
        raise SystemExit(f"--like folder has no corpus.jsonl: {src}")
    questions = load_stark_questions(args.name, args.split)
    random.Random(args.seed).shuffle(questions)
    if args.max_queries:
        questions = questions[: args.max_queries]
    os.makedirs(args.out, exist_ok=True)
    save_jsonl(os.path.join(args.out, "queries.jsonl"),
               [{"qid": str(qid), "query": q, "answers": [str(a) for a in ans]} for q, qid, ans in questions])
    linked = []
    for name in os.listdir(src):
        if name == "corpus.jsonl" or name.startswith("rows_") or name.startswith("cols_"):
            dst = os.path.join(args.out, name)
            if not os.path.exists(dst):
                os.symlink(os.path.join(src, name), dst)
                linked.append(name)
    print(f"wrote {len(questions)} '{args.split}' questions to {args.out}/queries.jsonl; "
          f"linked from {src}: {linked}")


# ---------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("make-toy")
    t.add_argument("--out", default="toy")

    s = sub.add_parser("prepare-stark")
    s.add_argument("--name", choices=["amazon", "mag", "prime"], default="amazon")
    s.add_argument("--split", default="test")
    s.add_argument("--out", required=True)
    s.add_argument("--max-queries", type=int, default=300)
    s.add_argument("--sample-rows", type=int, default=0, help="0 = full corpus")
    s.add_argument("--fields", default="", help="comma-separated attribute names to keep")
    s.add_argument("--max-chars", type=int, default=3000, help="character cap per field (0 = no cap)")
    s.add_argument("--no-relations", dest="relations", action="store_false",
                   help="do not turn relation types (brand, parent/child, ...) into columns")
    s.add_argument("--max-neighbors", type=int, default=20, help="linked items listed per relation (0 = all, as in mFAR)")
    s.add_argument("--no-flatten", dest="flatten_dicts", action="store_false",
                   help="keep nested attributes (e.g. Prime 'details') as one column")
    s.add_argument("--seed", type=int, default=0)

    pq = sub.add_parser("prepare-queries")
    pq.add_argument("--name", choices=["amazon", "mag", "prime"], required=True)
    pq.add_argument("--split", default="val")
    pq.add_argument("--like", required=True, help="existing prepared folder with the same table")
    pq.add_argument("--out", required=True)
    pq.add_argument("--max-queries", type=int, default=300)
    pq.add_argument("--seed", type=int, default=0)

    r = sub.add_parser("run")
    r.add_argument("--data", required=True)
    r.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="sentence-transformers model name, or 'hash' for a smoke test")
    r.add_argument("--query-prefix", default="", help="e.g. 'query: ' for E5 models")
    r.add_argument("--doc-prefix", default="", help="e.g. 'passage: ' for E5 models")
    r.add_argument("--cross-encoder", default="", help="e.g. cross-encoder/ms-marco-MiniLM-L-6-v2")
    r.add_argument("--ce-budget", type=int, default=100, help="candidates read by the CE")
    r.add_argument("--ce-batch-size", type=int, default=32)
    r.add_argument("--ce-fp16", action="store_true", help="run the cross-encoder in half precision")
    r.add_argument("--columns", default="", help="comma-separated; default = all fields")
    r.add_argument("--table-name", default="", help="prefix for whole-row text (stage 1)")
    r.add_argument("--keep-column-order", dest="short_first", action="store_false",
                   help="do not reorder columns short-first in the whole-row text")
    r.add_argument("--no-names", dest="use_names", action="store_false",
                   help="embed values only (ablation of 'column: value')")
    r.add_argument("--top-n", type=int, default=100, help="rows kept by each retriever list")
    r.add_argument("--rrf-k", type=int, default=60, help="RRF rank constant")
    r.add_argument("--bm25-weight", type=float, default=0.5,
                   help="keyword weight w in the hybrid combination score (fixed, not tuned)")
    r.add_argument("--gate-tau", type=float, default=0.5,
                   help="per-position gate: keep BM25 order while score >= tau * top BM25 score")
    r.add_argument("--gate-margin", type=float, default=0.1,
                   help="per-question gate: use BM25 if (top1 - top2) / top1 >= margin")
    r.add_argument("--gliner", action="store_true",
                   help="GLiNER+BM25: tag question phrases, bonus for BM25 top-K rows whose own column contains them")
    r.add_argument("--gl-model", default="urchade/gliner_medium-v2.1")
    r.add_argument("--gl-threshold", type=float, default=0.5)
    r.add_argument("--gl-dataset", default="auto", choices=["auto", "amazon", "mag", "prime"])
    r.add_argument("--gl-map", default="", help="JSON file {label: [columns]} replacing the built-in map")
    r.add_argument("--gl-ceiling", action="store_true",
                   help="ceiling of any tag-based policy + two design policies (use on the validation split)")
    r.add_argument("--timing", action="store_true", help="report milliseconds per question for each step")
    r.add_argument("--timing-n", type=int, default=50, help="questions used for encoder/GLiNER timing")
    r.add_argument("--selective", action="store_true",
                   help="selective reranking: cross-encoder only on questions a selector flags (needs --gliner --gl-ceiling --cross-encoder)")
    r.add_argument("--sel-mode", default="stem|sbert 0.7",
                   help="tag matching mode used by the alarm (fixed in advance)")
    r.add_argument("--gl-taus", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8],
                   help="SBERT similarity thresholds tried in the --gl-ceiling matching sweep")
    r.add_argument("--gl-sweep", action="store_true", help="also report lambda 0.5 and 2.0 (sensitivity only)")
    r.add_argument("--combo-new", action="store_true",
                   help="COMBO-NEW: query-level column relevance (names or content), decaying weighted sum, whole table, no RRF")
    r.add_argument("--cn-k", type=int, default=5, help="columns used by COMBO-NEW")
    r.add_argument("--cn-pool", choices=["table", "bm25"], default="table",
                   help="COMBO-NEW candidates: the whole table, or re-order BM25's top-K only")
    r.add_argument("--predictors", action="store_true",
                   help="compare confidence signals for the per-question gate (AUC + quantile gates + oracle)")
    r.add_argument("--gate-sweep", action="store_true",
                   help="also report other thresholds (sensitivity analysis, not for model selection)")
    r.add_argument("--exact-ablation", action="store_true",
                   help="also score combinations by embedding their joined text (slower)")
    r.add_argument("--diagnose", type=int, default=0,
                   help="write up to N cases where BM25 is right at rank 1 and your method is wrong")
    r.add_argument("--tag", default="", help="suffix for result files, e.g. --tag v2 -> results_v2.json")
    r.add_argument("--bm25-only", action="store_true",
                   help="quick data check: evaluate whole-row BM25 only (no embeddings, no GPU)")
    r.add_argument("--no-col-filter", dest="col_filter", action="store_false",
                   help="do not add the column filter (third candidate list)")
    r.add_argument("--col-filter-k", type=int, default=100, help="rows added by the column filter")
    r.add_argument("--max-combo", type=int, default=3, help="max columns per combination")
    r.add_argument("--prefilter-cols", type=int, default=6,
                   help="build combos only from the M most query-relevant columns of a row")
    r.add_argument("--size-penalty", type=float, default=0.01)
    r.add_argument("--alphas", default="0,0.3,0.5", help="weight of the RRF score when fused with combo")
    r.add_argument("--freeze-alpha", type=float, default=0.3,
                   help="the ONE alpha used as the main method and for CE fusion (fixed in advance)")
    r.add_argument("--ce-mix", type=float, default=0.5,
                   help="weight of the CE score when averaged with the combo score (fixed, not tuned)")
    r.add_argument("--perm", type=int, default=10000, help="permutations for significance tests")
    r.add_argument("--max-queries", type=int, default=0)
    r.add_argument("--batch-size", type=int, default=64)
    r.add_argument("--max-seq-len", type=int, default=512)
    r.add_argument("--device", default=None)
    r.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    {"make-toy": cmd_make_toy, "prepare-stark": cmd_prepare_stark,
     "prepare-queries": cmd_prepare_queries, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
