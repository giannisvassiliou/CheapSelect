# CheapSelect

Code, pre-registrations and per-query results for the paper

> **When Does Label-Free Selective Re-Ranking Pay Off? Experimental Study of Cheap Proxies for Expensive Re-Rankers**
> Giannis Vassiliou (Hellenic Mediterranean University) and Haridimos Kondylakis (FORTH-ICS & University of Crete).
> Submitted to PVLDB Vol. 20, Experiment, Analysis & Benchmark track.

Neural re-ranking improves retrieval, but it is expensive. Skipping it for queries that do not need it is a per-query cost–quality decision. The study measures when cheap, label-free signals from the first stage (WIG, NQC, the BM25 score margin, and others) can choose which queries to skip without losing more than a tolerance δ of MRR@10 or nDCG@10. It covers:

- **Main study:** 9 collections (STaRK Amazon, MAG and Prime; DBpedia-Entity; BEIR SciFact, NFCorpus, FiQA, ArguAna and SciDocs) × 3 re-rankers (MiniLM-L6, bge-reranker-v2-m3, monoT5-3B), giving 27 pairs.
- **Variants:** dense and hybrid first stages, and an 8B LLM re-ranker (Qwen3-Reranker-8B).
- **Two pre-registered held-out tests** of the practitioner checklist, each on 15 further pairs from collections never used to design it.

Every per-query result is released, so all tables except the timing measurements can be recomputed on a CPU in minutes.

**Results archive.** The results (per-query files, analysis tables, run logs and checksums; 374 files) are not in the repository tree. They are attached to the release as one file: download `CheapSelect_results.zip` from [release v1.0](https://github.com/giannisvassiliou/CheapSelect/releases/tag/v1.0) and unzip it in the repository folder, which creates `results/`.

---

## Contents

| Path | What it is |
|---|---|
| `README.md`, `LICENSE`, `CITATION.cff` | this file, MIT license, citation |
| `requirements.txt`, `requirements.lock` | Python packages; exact versions used for the paper |
| `src/` | all code; run the scripts from this folder (or set `WORKDIR`) |
| `prereg/` | pre-registration files, written before the corresponding runs |
| `figures/` | scripts and data for the figures |
| `tools/collect_results.sh` | copies per-query files, tables and logs from a working folder into `results/` |
| `results/` | **not in the tree**: unzip `CheapSelect_results.zip` from the v1.0 release |

**`src/`: pipeline and data**

| File | Purpose |
|---|---|
| `combo_search44.py` | retrieval and re-ranking pipeline: BM25, dense or hybrid first stage; re-rankers; selector signals; per-query output |
| `combo_search33/34/37/38/42/43.py` | earlier versions of the same pipeline that produced some of the paper's runs (see "Provenance") |
| `combo_search44.py prepare-stark`, `prepare-queries` | STaRK Amazon, MAG and Prime conversion (subcommands of the pipeline) |
| `prepare_beir.py` | DBpedia-Entity and all BEIR collections (including Climate-FEVER, HotpotQA, NQ, FEVER) |
| `fetch_mteb47.py` | CQADupStack forums from their MTEB copy on Hugging Face (removes duplicate judgements); `prepare_beir.py` then converts them |

**`src/`: analyses**

| File | Purpose | Paper |
|---|---|---|
| `common46.py` | loading per-query files, selector scores, budget helpers | all |
| `budgets46.py` | smallest non-inferior budgets per selector (point and bootstrap) | budget tables |
| `calib46.py` | calibration with 100–500 labels: point, UCB95, UCB-simul, LTT (Hoeffding, empirical Bernstein), CRC; random baselines; equal-label comparison | calibration tables |
| `groups46.py` | loss per query group (query type, tag alarm, query length) | group table |
| `paired46.py` | re-rankers compared on the same query ids | LLM table |
| `timing46.py` | first-stage, selector, re-ranker and router costs; calibrated policies end to end (GPU) | timing audit |
| `why42.py` | enrichment and helped/hurt decomposition of skipped queries | Section 5.2 |
| `router41.py`, `preret41.py`, `genc45.py` | supervised routers, pre-retrieval proxies, re-implemented 3-class router | router section |
| `cascade43.py`, `regime43.py`, `ndcg43.py` | cascades, regime statistics, nDCG@10 budgets | Section 5 |
| `calib42.py`, `calib43.py`, `latency42.py`, `lat45.py`, `sel_stats.py`, `sel_stats40.py`, `qpp_gain_corr.py` | earlier calibration, latency and statistics scripts called by the run scripts | |
| `holdout47.py` | pre-registered test 1 of the checklist (v1) | Section 6.2 |
| `explore47.py` | exploratory follow-up to test 1 (not pre-registered; included for transparency) | Section 6.3 |
| `holdout48.py` | pre-registered test 2 of the revised checklist (v2) | Section 6.3 |
| `run_*.sh` | run scripts, in the order of "Reproducing the paper" | |

**`prereg/`**

| File | Content |
|---|---|
| `FROZEN_dbpedia.txt` | hypothesis, label mapping and primary test for DBpedia-Entity (tag alarm) |
| `FROZEN_holdout47.txt` | checklist v1: rule, hypotheses H1–H4, md5 of the decision code |
| `FROZEN_holdout48.txt` | checklist v2: rule, hypotheses H1′–H4′, md5 of the decision code |

**`results/`** (from `CheapSelect_results.zip`)

| Path | Content |
|---|---|
| `per_query/<collection>/selective_pq_<tag>.jsonl` | one line per test query (format below) |
| `tables/` | analysis outputs: `b46`, `c46`, `l46`, `g46`, `p46`, `t46` (main study); `h47*`, `b47*` (held-out test 1); `e47*` (exploratory); `h48*`, `b48*` (held-out test 2); earlier statistics |
| `logs/` | progress logs of the run scripts |
| `development/` | validation-split runs on STaRK made while designing the tag alarm, before the frozen test-split runs (tags `final`, `final2`); not used for any reported number |
| `MANIFEST.md5` | checksums of everything under `results/` |

---

## Quick check without a GPU

Everything below needs only the per-query files and a CPU.

```bash
git clone https://github.com/giannisvassiliou/CheapSelect.git
cd CheapSelect
git checkout v1.0
```

Download `CheapSelect_results.zip` from the
[v1.0 release](https://github.com/giannisvassiliou/CheapSelect/releases/tag/v1.0)
into this folder, then:

```bash
unzip CheapSelect_results.zip      # creates results/
(cd results && md5sum -c --quiet MANIFEST.md5) && echo OK
pip install numpy scipy
cd src
# the scripts address collections as <folder>:<tag>
ln -s ../results/per_query/* .
```

**1. The decision code is the code that was pre-registered.**
Compare the two lines with the combined md5 in
`prereg/FROZEN_holdout47.txt` and `prereg/FROZEN_holdout48.txt`:

```bash
md5sum holdout47.py calib46.py common46.py | md5sum
md5sum holdout48.py calib46.py common46.py | md5sum
```

**2. Second held-out test** (about 15 minutes; same output as
`results/tables/h48.md`):

```bash
NEW="beir_cqa_physics beir_cqa_unix beir_cqa_tex
     beir_cqa_programmers beir_fever"
OLD="beir_cqa_english beir_cqa_gaming beir_climate-fever
     beir_hotpotqa beir_nq"
BEIR="beir_scifact beir_nfcorpus beir_fiqa
      beir_arguana beir_scidocs"
STARK="amazon_testfull mag_testfull prime_testfull"
RR="mini bge monot5"

H2=""; for d in $NEW; do for r in $RR; do
  H2="$H2 $d:h48$r"; done; done
H1=""; for d in $OLD; do for r in $RR; do
  H1="$H1 $d:h47$r"; done; done
IN=""
for d in $STARK; do IN="$IN $d:full $d:mini $d:monot5"; done
IN="$IN dbpedia_test:dbp dbpedia_test:mini"
IN="$IN dbpedia_test:monot5"
for d in $BEIR; do IN="$IN $d:bge $d:mini $d:monot5"; done

python holdout48.py $H2 --ref47 $H1 --insample $IN \
  --out h48_check
```

**3. Budgets and calibration** for the 27 study pairs:

```bash
python budgets46.py $IN --out b46_check
python calib46.py $IN --out c46_check
```

All analyses use fixed seeds (splits: seed 42), so their outputs match the released ones exactly.

---

## Setup (full reproduction)

- Python 3.10, one CUDA GPU. The paper used A100 40 GB; monoT5-3B in fp16 needs about 12 GB, and Qwen3-Reranker-8B in bf16 about 20 GB.
- `pip install -r requirements.txt`. The exact versions are in `requirements.lock`.
- Disk: about 150 GB for all corpora with their cached embeddings. The largest are DBpedia-Entity, Climate-FEVER, FEVER and HotpotQA, at 4.6–5.4M records each.
- The run scripts take these environment variables:

  | Variable | Meaning | Default |
  |---|---|---|
  | `WORKDIR` | working folder | the script's folder |
  | `VENV` | virtual environment to activate | none |
  | `GPU` | value for `CUDA_VISIBLE_DEVICES` | the value in each script |
  | `HF_LOCAL` | local copy of large models | `/tmp/hf_models` |

## Data

Nothing is redistributed. Every collection is downloaded from its
public source and converted to two JSONL files, one folder per
collection:

- `corpus.jsonl`: `{"id", "fields": {column: text}}`
- `queries.jsonl`: `{"qid", "query", "answers", "qtype"}`

Three converters cover all 19 collections:

| Collections | Source | Converter |
|---|---|---|
| STaRK Amazon, MAG, Prime | `stark-qa` package | `combo_search44.py prepare-stark` and `prepare-queries` |
| DBpedia-Entity v2; BEIR SciFact, NFCorpus, FiQA, ArguAna, SciDocs; Climate-FEVER, HotpotQA, NQ, FEVER | BEIR on Hugging Face (BEIR zip as fallback) | `prepare_beir.py` |
| CQADupStack English, Gaming, Physics, Unix, TeX, Programmers | MTEB copy on Hugging Face | `fetch_mteb47.py`, then `prepare_beir.py` |

The run scripts call these converters themselves. To prepare a
collection by hand, run the commands below from `src/`.

**STaRK** (full corpus, test split; the table is built once, then
the test questions are written next to it):

```bash
python combo_search44.py prepare-stark --name amazon \
  --split test --max-queries 0 --out amazon_paper
python combo_search44.py prepare-stark --name mag \
  --split test --max-queries 0 --out mag_paper
python combo_search44.py prepare-stark --name prime \
  --split test --max-queries 0 --out prime_full
for p in "amazon amazon_paper" "mag mag_paper" \
         "prime prime_full"; do
  set -- $p
  python combo_search44.py prepare-queries --name $1 \
    --split test --max-queries 0 --like $2 \
    --out $1_testfull
done
```

The folder names are the ones the run scripts expect.
Defaults of `prepare-stark`: full corpus, textual attributes
capped at 3,000 characters, and up to 20 linked items per relation
type as extra columns. The columns used for each collection are
listed in `run_full.sh`.

**DBpedia-Entity and BEIR:**

```bash
python prepare_beir.py --split test --out dbpedia_test
for d in scifact nfcorpus fiqa arguana scidocs \
         climate-fever hotpotqa nq fever; do
  python prepare_beir.py --name $d --split test \
    --out beir_$d
done
```

**CQADupStack:**

```bash
for f in english gaming physics unix tex programmers; do
  python fetch_mteb47.py mteb/cqadupstack-$f \
    cqadupstack/$f
  python prepare_beir.py --name cqadupstack-$f \
    --beir-dir cqadupstack/$f --split test \
    --out beir_cqa_$f
done
```

The MTEB copy of CQADupStack ships every judgement twice
(`qrels/test.jsonl` and `qrels/test.tsv`). `fetch_mteb47.py` reads
one file and drops duplicate pairs. Duplicates leave MRR unchanged
but lower nDCG@10. The two held-out-1 forums were first run with
duplicates and then re-run after de-duplication; only the re-runs
are used.

## Reproducing the paper

Run the scripts from `src/`, each inside `tmux`; each writes a `run_*_progress.log`. Times are for one A100 and are given where they were recorded.

| Step | Script | Produces | Paper | Time |
|---|---|---|---|---|
| 1 | `run_full.sh` | STaRK × bge (tag `full`) | main study |  |
| 2 | `run_dbpedia.sh` | DBpedia-Entity × bge (`dbp`), pre-registered tag-alarm test (`prereg/FROZEN_dbpedia.txt`) | main study, Sec. 4 |  |
| 3 | `run_rerank2.sh` | STaRK + DBpedia × MiniLM (`mini`) | main study |  |
| 4 | `run_rr_all38.sh` | STaRK + DBpedia × monoT5 (`monot5`) | main study |  |
| 5 | `run_beir_all.sh`, `run_beir_all38.sh` | BEIR × bge, MiniLM, monoT5 | main study |  |
| 6 | `run_review42.sh` | dense and hybrid first stages (`bgedense`, `bgehybrid`) | first-stage table | ~4 h |
| 7 | `run_review43.sh` | nDCG@10 and re-ranker features (`bge43`, `mini43`); cascades; regimes; relative tolerance | Sec. 5 | ~3 h (GPU part) |
| 8 | `run_llm44.sh` | Qwen3-Reranker-8B on fixed samples (`qwen8b`) | LLM table | ~10 h |
| 9 | `run_extra41.sh`, `run_genc45.sh` | pre-retrieval proxies, supervised routers, re-implemented 3-class router | router section | ~2 h (3-class router) |
| 10 | `run_why42.sh` | enrichment, helped/hurt decomposition | Sec. 5.2 | CPU, 1 min |
| 11 | `run_review46.sh` | budgets, calibration, equal-label comparison, groups, paired re-rankers (step 1, CPU); timing audit (step 2, GPU) | budget, calibration, group, cost and timing tables |  |
| 12 | `run_holdout47.sh` | pre-registered test 1 (checklist v1) | Sec. 6.2 | ~14 h |
| 13 | `run_holdout48.sh` | pre-registered test 2 (checklist v2) | Sec. 6.3 | ~8 h |

After the runs, collect everything into the repository:

```bash
bash tools/collect_results.sh /path/to/workdir
# the archive attached to the release:
zip -qr CheapSelect_results.zip results
```

### Pre-registered tests (steps 12 and 13)

Each held-out script writes `FROZEN_holdout4x.txt` in its step 0, before any run. That file records:
- the collections;
- the decision rule;
- the hypotheses, with what counts as failure;
- the md5 of the decision code (`holdout4x.py`, `calib46.py`, `common46.py`).

Step 3 refuses to run if that code has changed. To check our runs, copy `prereg/FROZEN_holdout47.txt` and `prereg/FROZEN_holdout48.txt` into the working folder before starting. Step 0 will then verify the code instead of writing a new file.

**What changed after freezing:**
- **Code-comment wording, not the decision code.** Up to three comment lines in each `combo_search*.py` file were reworded for publication, and the path, environment and GPU settings of the run scripts were turned into variables. As a result, the md5 of `combo_search44.py` listed in the FROZEN files for information no longer matches the published file. The decision code is byte-identical (checked in "Quick check").
- **A data correction:** the CQADupStack duplicate judgements described under "Data".

`explore47.py` was written **after** seeing the results of test 1 and is not pre-registered. It is included because it motivated checklist v2, which was then frozen and tested on new collections (test 2).

## Per-query file format

`selective_pq_<tag>.jsonl` has one JSON object per test query, in pipeline order. The fields used by the analyses are:

| Field | Content |
|---|---|
| `qid`, `qid_str` | query id |
| `qtype` | query type (DBpedia-Entity categories; empty elsewhere) |
| `bm` | first-stage metrics: `MRR`, `Hit@1`, `Hit@5`, `R@20` |
| `ce` | metrics after re-ranking the first-stage top 100 |
| `ndcg` | `{bm, ce}`: binary nDCG@10 (tags 43, 44, 47, 48) |
| `qpp` | label-free selector scores: `WIG-like`, `NQC-like spread`, `top5 vs rest`, `term coverage top1` |
| `margin` | BM25 score margin |
| `alarm` | GLiNER tag alarm |
| `cef` (43+) | re-ranker features used by cascades: `top1`, `margin`, `agree_top1`, `rho10`, `std` |

Gain is `g = ce.MRR − bm.MRR`. A selector re-ranks low scores first. Files named `selective_pq_<tag>ndcg.jsonl` hold the same queries with binary nDCG@10 in place of MRR, for the statistics scripts that read the `MRR` field.

## Provenance of the runs

The pipeline grew during the study. Each run of the paper was made with the version listed below. All versions are included, so every run can be repeated exactly. New runs should use `combo_search44.py`, which has every option of the earlier versions.

| Tag(s) | Collections | Pipeline version | Run script |
|---|---|---|---|
| `full` | STaRK × bge | `combo_search33.py` | `run_full.sh` |
| `dbp` | DBpedia-Entity × bge | `combo_search34.py` | `run_dbpedia.sh` |
| `mini` | STaRK + DBpedia × MiniLM | `combo_search34.py` | `run_rerank2.sh` |
| `bge`, `mini` | BEIR | `combo_search37.py` | `run_beir_all.sh` |
| `monot5` | STaRK, DBpedia, BEIR | `combo_search38.py` | `run_rr_all38.sh`, `run_beir_all38.sh` |
| `bgedense`, `bgehybrid` | all 9 | `combo_search42.py` | `run_review42.sh` |
| `bge43`, `mini43` | all 9 | `combo_search43.py` | `run_review43.sh` |
| `qwen8b` | 7 collections, fixed samples | `combo_search44.py` | `run_llm44.sh` |
| `h47*`, `h48*` | held-out collections | `combo_search44.py` | `run_holdout47.sh`, `run_holdout48.sh` |

## Notes

- Absolute latencies vary by up to 2× between sessions on shared GPUs. The paper compares policies within one measurement session (`timing46.py`, three repetitions) and reports ratios.
- Binary nDCG@10 (unjudged records count as non-relevant; ideal DCG over min(|rel|, 10)) is not comparable with published graded nDCG@10.
- Model checkpoints are loaded from Hugging Face by name: `cross-encoder/ms-marco-MiniLM-L-6-v2`, `BAAI/bge-reranker-v2-m3`, `castorini/monot5-3b-msmarco-10k`, `Qwen/Qwen3-Reranker-8B`, `sentence-transformers/all-MiniLM-L6-v2`, `urchade/gliner_medium-v2.1`.

## License and citation

Code: MIT (see `LICENSE`). The collections keep their own licenses. Only derived per-query numbers are released, never corpus text or queries.


