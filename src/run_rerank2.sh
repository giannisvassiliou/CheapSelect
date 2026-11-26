#!/bin/bash
# run_rerank2.sh: same frozen pipeline with a SECOND re-ranker (small MiniLM cross-encoder), all 4 test collections.
#   tmux new -s rr2   then:  bash run_rerank2.sh   then:  Ctrl+b, d        check: cat run_rerank2_progress.log
cd "${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}" || exit 1
[ -n "$VENV" ] && source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="${GPU:-7}"
export PYTHONUNBUFFERED=1
LOG=run_rerank2_progress.log
say() { echo "$(date '+%F %T')  $*" | tee -a "$LOG"; }

COLS_AMZ="title,brand,description,feature,review,qa,also_buy,also_view"
COLS_MAG2="title,abstract,author___affiliated_with___institution,paper___has_topic___field_of_study,paper___cites___paper"
COLS="name,details.alias,details.category,details.description,details.summary,details.summation,details.indication,details.mechanism_of_action,details.pharmacodynamics,details.mondo_definition,details.umls_description,details.orphanet_definition,details.orphanet_clinical_description,details.mayo_symptoms,details.mayo_causes,carrier,target,transporter,enzyme,expression absent,expression present,phenotype absent,phenotype present,off-label use,indication,contraindication,side effect,synergistic interaction,parent-child,linked to,associated with,interacts with,ppi"

RR2="cross-encoder/ms-marco-MiniLM-L-6-v2"
OPTS="--device cuda --batch-size 256 --no-col-filter --gliner --gl-ceiling --cross-encoder $RR2 --ce-fp16 --selective --timing --tag mini"

run() {  # $1 folder, $2 columns, $3 log
  say "run $1 with $RR2"
  python -u combo_search34.py run --data "$1" $OPTS --columns "$2" > "$3" 2>&1 && say "done $1" || say "FAILED $1 (see $3)"
}
say "START"
run prime_testfull  "$COLS"           mini_prime.log
run mag_testfull    "$COLS_MAG2"      mini_mag.log
run amazon_testfull "$COLS_AMZ"       mini_amazon.log
run dbpedia_test    "title,abstract"  mini_dbp.log

say "stats ..."
python sel_stats.py amazon_testfull mag_testfull prime_testfull --tag mini --out sel_stats_mini.md >> "$LOG" 2>&1
python sel_stats.py dbpedia_test --tag mini --out sel_stats_mini_dbp.md >> "$LOG" 2>&1
python qpp_gain_corr.py amazon_testfull:mini mag_testfull:mini prime_testfull:mini dbpedia_test:mini >> "$LOG" 2>&1
mv qpp_gain_corr.md qpp_gain_corr_mini.md 2>/dev/null
say "ALL DONE: sel_stats_mini.md, sel_stats_mini_dbp.md, qpp_gain_corr_mini.md"
