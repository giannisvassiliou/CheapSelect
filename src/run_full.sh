#!/bin/bash
# run_full.sh: frozen selective-reranking test on the FULL STaRK test sets, one dataset after another.
# Start it inside tmux, detach, and come back later:
#   tmux new -s full        then:  bash run_full.sh        then:  Ctrl+b, then d   (detach)
#   later:  tmux attach -t full      or:  tail -f run_full_progress.log

cd "${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}" || exit 1
[ -n "$VENV" ] && source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="${GPU:-7}"
export PYTHONUNBUFFERED=1

COLS_AMZ="title,brand,description,feature,review,qa,also_buy,also_view"
COLS_MAG2="title,abstract,author___affiliated_with___institution,paper___has_topic___field_of_study,paper___cites___paper"
COLS="name,details.alias,details.category,details.description,details.summary,details.summation,details.indication,details.mechanism_of_action,details.pharmacodynamics,details.mondo_definition,details.umls_description,details.orphanet_definition,details.orphanet_clinical_description,details.mayo_symptoms,details.mayo_causes,carrier,target,transporter,enzyme,expression absent,expression present,phenotype absent,phenotype present,off-label use,indication,contraindication,side effect,synergistic interaction,parent-child,linked to,associated with,interacts with,ppi"

OPTS="--device cuda --batch-size 256 --no-col-filter --gliner --gl-ceiling --cross-encoder BAAI/bge-reranker-v2-m3 --ce-fp16 --selective --timing --tag full"
LOG=run_full_progress.log
say() { echo "$(date '+%F %T')  $*" | tee -a "$LOG"; }

say "START"

# 1. question files for the full test sets (fast; no re-embedding)
for pair in "amazon amazon_paper amazon_testfull" "mag mag_paper mag_testfull" "prime prime_full prime_testfull"; do
  set -- $pair
  if [ ! -s "$3/queries.jsonl" ]; then
    say "prepare-queries $1 -> $3"
    python combo_search33.py prepare-queries --name "$1" --split test --max-queries 0 --like "$2" --out "$3" >> "$LOG" 2>&1 \
      || say "FAILED prepare-queries $1"
  fi
  say "$3: $(wc -l < "$3/queries.jsonl" 2>/dev/null) questions"
done

# 2. the three runs, one after another (a failure does not stop the next one)
run() {  # $1 = folder, $2 = columns, $3 = log name
  say "run $1 ..."
  python -u combo_search33.py run --data "$1" $OPTS --columns "$2" > "$3" 2>&1 \
    && say "done $1" || say "FAILED $1 (see $3)"
}
run prime_testfull  "$COLS"      full_prime.log
run mag_testfull    "$COLS_MAG2" full_mag.log
run amazon_testfull "$COLS_AMZ"  full_amazon.log

# 3. statistics
say "stats ..."
grep "tag alarm fires on" full_*.log | tee -a "$LOG"
python sel_stats.py amazon_testfull mag_testfull prime_testfull --tag full --out sel_stats_full.md >> "$LOG" 2>&1 \
  && say "ALL DONE: see sel_stats_full.md" || say "FAILED stats"
