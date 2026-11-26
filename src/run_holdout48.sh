#!/bin/bash
# run_holdout48.sh: SECOND pre-registered held-out test, of the revised checklist v2 (holdout48.py).
# Collections never used before: four more CQADupStack forums and FEVER (a different task: claim verification),
# same frozen pipeline (combo_search44.py), same three re-rankers, up to 1000 test queries each (seed 0).
#   STEP 0 (seconds)  pre-registration FROZEN_holdout48.txt, written BEFORE any run on these collections
#   STEP 1 (CPU)      data (CQADupStack via the MTEB copy on Hugging Face, FEVER via BEIR on Hugging Face)
#   STEP 2 (GPU)      runs (~7 h: FEVER ~4 h, forums ~45 min each); start it after run_holdout47.sh has finished
#   STEP 3 (CPU)      holdout48.py on the new pairs, with the first held-out set and the 27 pairs for comparison
#   start now:   tmux new -d -s h48 'GPU=7 bash run_holdout48.sh'        progress: cat run_holdout48_progress.log
cd "${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}" || exit 1
[ -n "$VENV" ] && source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="${GPU:-7}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STEPS="${STEPS:-0 1 2 3}"
LOG="run_holdout48_progress.log"
say() { echo "$(date '+%F %T')  $*" | tee -a "$LOG"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
for f in holdout48.py calib46.py common46.py combo_search44.py prepare_beir.py fetch_mteb47.py budgets46.py; do
  [ -f "$f" ] || { say "MISSING $f - copy it first"; exit 1; }
done
DATASETS="${DATASETS:-cqa_physics cqa_unix cqa_tex cqa_programmers fever}"
RERANKERS="${RERANKERS:-mini bge monot5}"
NQ_MAX=1000
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
code_md5() { md5sum holdout48.py calib46.py common46.py | md5sum | cut -c1-32; }

# ---------------- STEP 0: pre-registration
if has 0; then
  if [ -s FROZEN_holdout48.txt ]; then
    grep -q "$(code_md5)" FROZEN_holdout48.txt && say "0. FROZEN_holdout48.txt exists, code unchanged" \
      || { say "0. STOP: holdout48.py/calib46.py/common46.py differ from the frozen versions"; exit 1; }
  else
    cat > FROZEN_holdout48.txt <<EOF
PRE-REGISTRATION 2: revised checklist (v2) on a second held-out set
Written $(date '+%F %T %Z') BEFORE any run on these collections. Nothing below may change after seeing results.

Why a second test: v2 was designed after the first held-out test (FROZEN_holdout47.txt), where the fixed choice of WIG
left about 10 points of saving and the rule "rho >= 15 -> re-rank all" missed HotpotQA. Its success there is partly
circular; this test uses collections that played no part in designing v1 or v2.

Collections (none used before; up to $NQ_MAX test queries each, seed 0):
  CQADupStack physics, unix, tex, programmers (MTEB copy of the BEIR data); FEVER (BEIR), a different task family.
Re-rankers: MiniLM-L6, bge-reranker-v2-m3, monoT5-3B; pipeline combo_search44.py unchanged (BM25 top-100).

Decision rule v2 (holdout48.py): n = 200 labelled queries; certify every cheap selector (six proxies + tag alarm) and
analytic random on them with a one-sided normal bound at z = Phi^-1(1 - 0.05 / (selectors x 10 thresholds)); keep the
largest estimated saving; re-rank all if none skips anything. No regime rule. 200 splits, delta in {0.5, 1, 2} points,
MRR@10 and nDCG@10; correct = no violation and regret <= 5 points (as before).

Hypotheses (pooled over the cells of THIS test; FEVER is also reported on its own):
  H1 (safety)       v2 exceeds delta on the test queries in at most 5% of splits.
  H2 (saving)       v2's mean saving is at least v1's.
  H3 (quality)      v2's P(correct) is at least v1's.
  H4 (no safety cost) v2's violation rate is at most v1's + 1 point.
Each fails if its inequality does not hold. The first held-out set and the 27 study pairs are reported for comparison.

md5 of the decision code (holdout48.py, calib46.py, common46.py, combined):
$(code_md5)
$(md5sum holdout48.py calib46.py common46.py combo_search44.py)
EOF
    say "0. wrote FROZEN_holdout48.txt (e-mail it to yourself NOW, before step 2, for a timestamp)"
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
        python fetch_mteb47.py "mteb/cqadupstack-$sub" "cqadupstack/$sub" >> "$LOG" 2>&1 \
          && python prepare_beir.py --name "cqadupstack-$sub" --beir-dir "cqadupstack/$sub" --split test --out "$out" >> "$LOG" 2>&1 ;;
      *)
        python prepare_beir.py --name "$d" --split test --out "$out" >> "$LOG" 2>&1 ;;
    esac
    [ -s "$out/corpus.jsonl" ] && say "  $d: $(wc -l < $out/corpus.jsonl) records, $(wc -l < $out/queries.jsonl) queries" \
                               || say "  FAILED to prepare $d"
  done
  rm -rf cqadupstack
  say "STEP 1 DONE"
fi

# ---------------- STEP 2: runs (after the first held-out test has released the GPU)
if has 2; then
  [ -s FROZEN_holdout48.txt ] || { say "STOP: run STEP 0 first"; exit 1; }
  say "2. starting on GPU $CUDA_VISIBLE_DEVICES"
  for d in $DATASETS; do
    out="beir_${d}"
    [ -s "$out/corpus.jsonl" ] || { say "2. SKIP $d (not prepared)"; continue; }
    cols=$(beircols "$out")
    for r in $RERANKERS; do
      tag="h48${r}"
      [ -s "$out/selective_pq_${tag}.jsonl" ] && { say "2. $d $r already done"; continue; }
      say "2. run $d with $r (columns $cols)"
      python -u combo_search44.py run --data "$out" --columns "$cols" --gl-dataset dbpedia \
        --device cuda --batch-size 256 --no-col-filter --gliner --gl-ceiling \
        --cross-encoder "${RRNAME[$r]}" --ce-fp16 --ce-batch-size "${RRBS[$r]}" \
        --selective --ce-selective-only --max-queries "$NQ_MAX" --seed 0 --tag "$tag" > "h48_${d}_${r}.log" 2>&1 \
        && say "  done ($(wc -l < $out/selective_pq_${tag}.jsonl) queries)" || say "  FAILED (see h48_${d}_${r}.log)"
    done
  done
  say "STEP 2 DONE"
fi

# ---------------- STEP 3: analysis
if has 3; then
  grep -q "$(code_md5)" FROZEN_holdout48.txt || { say "STOP: decision code changed since pre-registration"; exit 1; }
  NEW=""; for d in $DATASETS; do for r in mini bge monot5; do
    [ -s "beir_${d}/selective_pq_h48${r}.jsonl" ] && NEW="$NEW beir_${d}:h48${r}"; done; done
  OLD=""; for d in cqa_english cqa_gaming climate-fever hotpotqa nq; do for r in mini bge monot5; do
    [ -s "beir_${d}/selective_pq_h47${r}.jsonl" ] && OLD="$OLD beir_${d}:h47${r}"; done; done
  BEIR="scifact nfcorpus fiqa arguana scidocs"
  IN=""; for t in "full dbp bge" "mini mini mini" "monot5 monot5 monot5"; do set -- $t
    for d in amazon_testfull mag_testfull prime_testfull; do IN="$IN $d:$1"; done; IN="$IN dbpedia_test:$2"
    for d in $BEIR; do IN="$IN beir_$d:$3"; done; done
  say "3. holdout48 on:$NEW"
  python holdout48.py $NEW --ref47 $OLD --insample $IN --out h48 > h48.log 2>&1 || say "  FAILED (see h48.log)"
  python budgets46.py $NEW --out b48_holdout > /dev/null 2>&1
  tar czf holdout48_results.tgz FROZEN_holdout48.txt h48.md h48_pairs.csv h48.log b48_holdout* run_holdout48_progress.log 2>/dev/null
  say "ALL DONE: results: holdout48_results.tgz"
fi
