#!/bin/bash
# run_holdout47.sh: held-out test of the practitioner checklist (reviewer: "never tested on data it wasn't built from").
# Five BEIR collections that played no part in the study, the SAME frozen pipeline (combo_search44.py, BM25 top-100,
# same signals), the same three re-rankers, a fixed sample of up to 1000 test queries each (seed 0).
#   STEP 0 (seconds): pre-registration file FROZEN_holdout47.txt with the md5 of the decision rule, written BEFORE any run
#   STEP 1 (CPU, ~1 h, ~15 GB download): prepare the data
#   STEP 2 (GPU, ~12-16 h sequential): one run per collection x re-ranker (index building on the 2.7-5.4M-record
#          corpora dominates the first run of each; the embeddings are cached for the next re-ranker)
#   STEP 3 (CPU, ~10 min): holdout47.py on the held-out pairs, and on the 27 in-sample pairs for comparison
#
#   simplest:   tmux new -s h47 'GPU=3 bash run_holdout47.sh'          progress: cat run_holdout47_progress.log
#   faster (3 GPUs, needs ~3x the RAM; check free -g first):
#     STEPS="0 1" bash run_holdout47.sh                                      (once, wait for it)
#     tmux new -s h47m 'STEPS=2 RERANKERS=mini   GPU=3 bash run_holdout47.sh'
#     tmux new -s h47b 'STEPS=2 RERANKERS=bge    GPU=6 bash run_holdout47.sh'
#     tmux new -s h47t 'STEPS=2 RERANKERS=monot5 GPU=7 bash run_holdout47.sh'
#     when all three say DONE:   STEPS=3 bash run_holdout47.sh
#   fewer collections:  DATASETS="cqa_english cqa_gaming nq" ...
cd "${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}" || exit 1
[ -n "$VENV" ] && source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="${GPU:-3}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STEPS="${STEPS:-0 1 2 3}"
LOG="run_holdout47_progress.log"
say() { echo "$(date '+%F %T')  $*" | tee -a "$LOG"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
for f in holdout47.py calib46.py common46.py combo_search44.py prepare_beir.py; do
  [ -f "$f" ] || { say "MISSING $f - copy it first"; exit 1; }
done

DATASETS="${DATASETS:-cqa_english cqa_gaming climate-fever hotpotqa nq}"
RERANKERS="${RERANKERS:-mini bge monot5}"
NQ_MAX="${NQ_MAX:-1000}"
declare -A RRNAME=( [bge]="BAAI/bge-reranker-v2-m3" [mini]="cross-encoder/ms-marco-MiniLM-L-6-v2"
                    [monot5]="castorini/monot5-3b-msmarco-10k" )
declare -A RRBS=( [bge]=32 [mini]=64 [monot5]=4 )
beircols() {
python - "$1" <<'EOF'
import json, sys
seen = {"title": 0, "abstract": 0}
for i, l in enumerate(open(sys.argv[1] + "/corpus.jsonl")):
    f = json.loads(l)["fields"]
    for k in seen:
        seen[k] += bool(f.get(k))
    if i > 20000:
        break
print(",".join(k for k, v in seen.items() if v > 0))
EOF
}

# ---------------- STEP 0: pre-registration (never overwritten)
if has 0; then
  if [ -s FROZEN_holdout47.txt ]; then
    say "0. FROZEN_holdout47.txt exists; checking that the decision rule has not changed"
    now=$(md5sum holdout47.py calib46.py common46.py | md5sum | cut -c1-32)
    grep -q "$now" FROZEN_holdout47.txt && say "  OK, unchanged" || { say "  STOP: holdout47.py/calib46.py/common46.py differ from the frozen versions"; exit 1; }
  else
    cat > FROZEN_holdout47.txt <<EOF
PRE-REGISTRATION: held-out test of the practitioner checklist
Written $(date '+%F %T %Z') BEFORE any run on these collections. Nothing below may change after seeing results.

Collections (BEIR test splits, none used to build the checklist; up to $NQ_MAX queries each, sampled with seed 0):
  $DATASETS
Re-rankers: MiniLM-L6, bge-reranker-v2-m3, monoT5-3B; pipeline combo_search44.py unchanged (BM25 top-100).

Decision rule (holdout47.py): n = 200 labelled queries; rho = mean gain on them / delta;
  rho <= 1 -> random share (analytic, UCB95);  rho >= 15 -> re-rank all;
  otherwise WIG certified with UCB95, kept only if its estimated saving beats random's.
  delta in {0.5, 1, 2} points, MRR@10 and nDCG@10; 200 splits; correct = no violation and regret <= 5 points.

Hypotheses and what counts as failure:
  H1 (safety)    the checklist's policy exceeds delta on the test queries in at most 10% of splits, pooled over the
                 held-out cells. Fails if above 10%.
  H2 (quality)   the checklist is correct (no violation, within 5 points of the best certified policy in hindsight)
                 in at least 75% of splits, pooled. Fails if below 75%.
  H3 (value of the regime step)  the checklist's pooled P(correct) is not lower than that of "WIG vs random without
                 the regime step" and of "always WIG". Reported either way.
  H4 (units)     the boundaries hold in units of delta: pooled over held-out cells, the best policy in hindsight is
                 random or none for rho <= 1, and selectors save less than 10% for rho >= 15. Reported per rho bucket.
The in-sample 27 pairs are evaluated with the same code for comparison only.

md5 of the decision code (holdout47.py, calib46.py, common46.py, combined):
$(md5sum holdout47.py calib46.py common46.py | md5sum | cut -c1-32)
$(md5sum holdout47.py calib46.py common46.py combo_search44.py)
EOF
    say "0. wrote FROZEN_holdout47.txt (commit it or e-mail it to yourself NOW, before step 2, for a timestamp)"
  fi
fi

# ---------------- STEP 1: data
if has 1; then
  for d in $DATASETS; do
    out="beir_${d}"
    [ -s "$out/corpus.jsonl" ] && { say "1. $d already prepared"; continue; }
    say "1. prepare $d"
    case "$d" in
      cqa_*)
        sub="${d#cqa_}"
        if [ ! -d cqadupstack/"$sub" ]; then
          [ -s cqadupstack.zip ] || wget -q "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip"
          unzip -q -o cqadupstack.zip "cqadupstack/$sub/*"
        fi
        python prepare_beir.py --name "cqadupstack-$sub" --beir-dir "cqadupstack/$sub" --split test --out "$out" >> "$LOG" 2>&1 ;;
      *)
        python prepare_beir.py --name "$d" --split test --out "$out" >> "$LOG" 2>&1
        if [ ! -s "$out/corpus.jsonl" ]; then
          say "  Hugging Face failed for $d; trying the BEIR zip"
          wget -q "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/${d}.zip" \
            && unzip -q -o "${d}.zip" && python prepare_beir.py --name "$d" --beir-dir "$d" --split test --out "$out" >> "$LOG" 2>&1
        fi ;;
    esac
    [ -s "$out/corpus.jsonl" ] && say "  $d: $(wc -l < $out/corpus.jsonl) records, $(wc -l < $out/queries.jsonl) queries, columns $(beircols $out)" \
                               || say "  FAILED to prepare $d"
  done
  say "STEP 1 DONE (disk: $(du -sh beir_* 2>/dev/null | tail -n 5 | tr '\n' ' '))"
fi

# ---------------- STEP 2: runs (same options as the in-sample BEIR runs, plus the fixed query sample)
if has 2; then
  [ -s FROZEN_holdout47.txt ] || { say "STOP: run STEP 0 first (pre-registration)"; exit 1; }
  for d in $DATASETS; do
    out="beir_${d}"
    [ -s "$out/corpus.jsonl" ] || { say "2. SKIP $d (not prepared)"; continue; }
    cols=$(beircols "$out")
    for r in $RERANKERS; do
      tag="h47${r}"
      [ -s "$out/selective_pq_${tag}.jsonl" ] && { say "2. $d $r already done"; continue; }
      say "2. run $d with $r on GPU $CUDA_VISIBLE_DEVICES (columns $cols)"
      python -u combo_search44.py run --data "$out" --columns "$cols" --gl-dataset dbpedia \
        --device cuda --batch-size 256 --no-col-filter --gliner --gl-ceiling \
        --cross-encoder "${RRNAME[$r]}" --ce-fp16 --ce-batch-size "${RRBS[$r]}" \
        --selective --ce-selective-only --max-queries "$NQ_MAX" --seed 0 --tag "$tag" > "h47_${d}_${r}.log" 2>&1 \
        && say "  done ($(wc -l < $out/selective_pq_${tag}.jsonl) queries)" || say "  FAILED (see h47_${d}_${r}.log)"
    done
  done
  say "STEP 2 DONE for re-rankers: $RERANKERS"
fi

# ---------------- STEP 3: analysis
if has 3; then
  now=$(md5sum holdout47.py calib46.py common46.py | md5sum | cut -c1-32)
  grep -q "$now" FROZEN_holdout47.txt || { say "STOP: decision code changed since pre-registration"; exit 1; }
  HO=""; for d in $DATASETS; do for r in mini bge monot5; do
    [ -s "beir_${d}/selective_pq_h47${r}.jsonl" ] && HO="$HO beir_${d}:h47${r}"; done; done
  BEIR="scifact nfcorpus fiqa arguana scidocs"
  IN=""; for t in "full dbp bge" "mini mini mini" "monot5 monot5 monot5"; do set -- $t
    for d in amazon_testfull mag_testfull prime_testfull; do IN="$IN $d:$1"; done; IN="$IN dbpedia_test:$2"
    for d in $BEIR; do IN="$IN beir_$d:$3"; done; done
  say "3. holdout47 on:$HO"
  python holdout47.py $HO --insample $IN --metrics MRR ndcg --out h47 > h47.log 2>&1 || say "  FAILED (see h47.log)"
  # per-pair budgets on the held-out pairs (same table as the paper's main budget table)
  python budgets46.py $HO --out b47_holdout > /dev/null 2>&1 || say "  budgets46 FAILED (optional)"
  tar czf holdout47_results.tgz FROZEN_holdout47.txt h47.md h47_pairs.csv h47_buckets.csv h47.log b47_holdout* \
      run_holdout47_progress.log $(ls h47_*_*.log 2>/dev/null) 2>/dev/null
  say "ALL DONE: results: holdout47_results.tgz"
fi
