# When Does Label-Free Selective Re-Ranking Pay Off?

Code, pre-registrations and per-query results for the paper

> **When Does Label-Free Selective Re-Ranking Pay Off? Experimental Study of Cheap Proxies for Expensive Re-Rankers**
> Giannis Vassiliou (Hellenic Mediterranean University) and Haridimos Kondylakis (FORTH-ICS & University of Crete).
> Submitted to PVLDB Vol. 20, Experiment, Analysis & Benchmark track.

Neural re-ranking improves retrieval, but it is expensive. Skipping it for queries that do not need it is a per-query cost–quality decision. The study measures when cheap, label-free signals from the first stage (WIG, NQC, the BM25 score margin, and others) can choose which queries to skip without losing more than a tolerance δ of MRR@10 or nDCG@10. It covers:

- **Main study:** 9 collections (STaRK Amazon, MAG and Prime; DBpedia-Entity; BEIR SciFact, NFCorpus, FiQA, ArguAna and SciDocs) × 3 re-rankers (MiniLM-L6, bge-reranker-v2-m3, monoT5-3B), giving 27 pairs.
- **Variants:** dense and hybrid first stages, and an 8B LLM re-ranker (Qwen3-Reranker-8B).
- **Two pre-registered held-out tests** of the practitioner checklist, each on 15 further pairs from collections never used to design it.

Every per-query result is released, so all tables except the timing measurements can be recomputed on a CPU in minutes.

---

## Contents

```
README.md            this file
requirements.txt     Python packages (exact versions: requirements.lock)
src/                 all code; the run scripts expect to be started from this folder (or set WORKDIR)
  combo_search44.py    retrieval + re-ranking pipeline (BM25 / dense / hybrid first stage, re-rankers, selector
                       signals, per-query output); combo_search33/34/37/38/42/43.py are the earlier versions that
                       produced some of the paper's runs (see "Provenance")
  prepare_beir.py      BEIR / DBpedia-Entity download and conversion
  fetch_mteb47.py      CQADupStack from its MTEB copy on Hugging Face (de-duplicates the judgements)
  common46.py          loading per-query files; selector scores; budget helpers
  budgets46.py         smallest non-inferior budgets (point and bootstrap) per selector       -> budget tables
  calib46.py           calibration with 100-500 labels: point, UCB95, UCB-simul, LTT (Hoeffding,
                       empirical Bernstein), CRC; random baselines; equal-label comparison       -> calibration tables
  groups46.py          loss per query group (query type, tag alarm, query length)              -> group table
  paired46.py          re-rankers compared on the same query ids                               -> LLM table
  timing46.py          first-stage, selector, re-ranker and router costs; calibrated policies end to end (GPU)
  why42.py             enrichment and helped/hurt decomposition of skipped queries             -> Section 5.2
  router41.py, preret41.py, genc45.py     supervised routers, pre-retrieval proxies, re-implemented 3-class router
  cascade43.py, regime43.py, ndcg43.py, calib43.py, latency42.py, lat45.py, sel_stats40.py, qpp_gain_corr.py
                       cascades, regime statistics, nDCG@10 budgets, earlier calibration, latency, statistics
  holdout47.py         pre-registered test 1 of the checklist (v1)
  explore47.py         exploratory follow-up to test 1 (not pre-registered; shown for transparency)
  holdout48.py         pre-registered test 2 of the revised checklist (v2)
  run_*.sh             the run scripts, in the order of "Reproducing the paper"
prereg/              pre-registration files written before the corresponding runs
  FROZEN_dbpedia.txt     hypothesis, label mapping and primary test for DBpedia-Entity (tag alarm)
  FROZEN_holdout47.txt   checklist v1: rule, hypotheses H1-H4, md5 of the decision code
  FROZEN_holdout48.txt   checklist v2: rule, hypotheses H1'-H4', md5 of the decision code
results/
  per_query/<collection>/selective_pq_<tag>.jsonl   one line per test query (format below)
  heldout1/            outputs of holdout47.py (h47.md, h47_pairs.csv, h47_buckets.csv) and budgets46.py
  heldout2/            outputs of holdout48.py (h48.md, h48_pairs.csv) and budgets46.py
  tables/, logs/       analysis outputs and run logs (filled by tools/collect_results.sh)
  MANIFEST.md5         checksums of everything under results/
figures/             scripts and data for the figures (curves_grid.py, gainfig.py, enrichfig.py)
tools/collect_results.sh   copies per-query files, tables and logs from a working directory into results/
```

---

## Quick check without a GPU

Everything below needs only the per-query files in `results/per_query/` and a CPU.

```bash
pip install numpy scipy
cd src
ln -s ../results/per_query/* .          # the scripts address collections as <folder>:<tag>

# 1. the decision code is the code that was pre-registered
md5sum holdout47.py calib46.py common46.py | md5sum   # = combined md5 in prereg/FROZEN_holdout47.txt
md5sum holdout48.py calib46.py common46.py | md5sum   # = combined md5 in prereg/FROZEN_holdout48.txt

# 2. second held-out test (about 15 minutes; output identical to results/heldout2/h48.md)
python holdout48.py beir_cqa_{physics,unix,tex,programmers}:h48{mini,bge,monot5} beir_fever:h48{mini,bge,monot5} \
  --ref47 beir_{cqa_english,cqa_gaming,climate-fever,hotpotqa,nq}:h47{mini,bge,monot5} \
  --insample {amazon,mag,prime}_testfull:full dbpedia_test:dbp beir_{scifact,nfcorpus,fiqa,arguana,scidocs}:bge \
             {amazon,mag,prime}_testfull:mini dbpedia_test:mini beir_{scifact,nfcorpus,fiqa,arguana,scidocs}:mini \
             {amazon,mag,prime}_testfull:monot5 dbpedia_test:monot5 beir_{scifact,nfcorpus,fiqa,arguana,scidocs}:monot5 \
  --out h48_check

# 3. budgets and calibration for the 27 pairs
python budgets46.py <the 27 pairs as above> --out b46_check
python calib46.py  <the 27 pairs as above> --out c46_check
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

Nothing is redistributed. Every corpus is downloaded from its public source and converted to two JSONL files:

- `corpus.jsonl`: `{"id", "fields": {column: text}}`
- `queries.jsonl`: `{"qid", "query", "answers", "qtype"}`

| Collections | Source | Command |
|---|---|---|
| STaRK Amazon, MAG, Prime (full test splits) | `stark-qa` | `python combo_search44.py prepare-stark --name {amazon,mag,prime} --out <folder> ...`, then `prepare-queries --split test --max-queries 0 --like <folder> --out {amazon,mag,prime}_testfull` (see `run_full.sh`) |
| DBpedia-Entity v2, test | BEIR on Hugging Face | `python prepare_beir.py --split test --out dbpedia_test` |
| BEIR SciFact, NFCorpus, FiQA, ArguAna, SciDocs | BEIR on Hugging Face (zip as fallback) | inside `run_beir_all*.sh` |
| Held-out 1: CQADupStack English, Gaming; Climate-FEVER; HotpotQA; NQ | MTEB copy (CQADupStack), BEIR | `run_holdout47.sh`, step 1 |
| Held-out 2: CQADupStack Physics, Unix, TeX, Programmers; FEVER | MTEB copy (CQADupStack), BEIR | `run_holdout48.sh`, step 1 |

The MTEB copy of CQADupStack ships every judgement twice (`qrels/test.jsonl` and `qrels/test.tsv`). `fetch_mteb47.py` reads one file and drops duplicate pairs. Duplicates leave MRR unchanged but lower nDCG@10. The two held-out-1 forums were first run with duplicates and then re-run after de-duplication; only the re-runs are used.

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

Gain is `g = ce.MRR − bm.MRR`. A selector re-ranks low scores first.

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

If you use the code or the per-query results, please cite the paper (see `CITATION.cff`; the DOI will be added on publication).
