#!/bin/bash
# run_review46.sh: computations for the yellow "Before submission" items (1-11) of the revised paper.
#   STEP 1 (CPU, ~30-60 min): budgets with b = 0 (item 2), calibration with separate design/certification, analytic random,
#          per-pair outputs (items 4, 5, 6), equal-label-budget comparison (item 10), group decomposition (item 7),
#          re-rankers on the Qwen query ids (item 8)
#   STEP 2 (GPU, ~1-2 h): clean repeated timings, title-only check, BERT router cost, calibrated policies end to end (items 3, 11)
#   tmux new -s r46 'GPU=6 bash run_review46.sh'          (or STEPS="1" / STEPS="2")
cd "${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}" || exit 1
[ -n "$VENV" ] && source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="${GPU:-6}"
export PYTHONUNBUFFERED=1
STEPS="${STEPS:-1 2}"
LOG="run_review46_progress.log"
say() { echo "$(date '+%F %T')  $*" | tee -a "$LOG"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
for f in common46.py budgets46.py calib46.py groups46.py paired46.py timing46.py combo_search44.py; do
  [ -f "$f" ] || { say "MISSING $f - copy it first"; exit 1; }
done
BEIR="scifact nfcorpus fiqa arguana scidocs"
pairs() { for d in amazon_testfull mag_testfull prime_testfull; do echo "$d:$1"; done; echo "dbpedia_test:$2"; for d in $BEIR; do echo "beir_$d:$3"; done; }
STK() { for d in amazon_testfull mag_testfull prime_testfull; do echo "$d:$1"; done; echo "dbpedia_test:$2"; }
BR()  { for d in $BEIR; do echo "beir_$d:$1"; done; }
QW="mag_testfull prime_testfull amazon_testfull dbpedia_test beir_scifact beir_nfcorpus beir_scidocs"
COLS_AMZ="title,brand,description,feature,review,qa,also_buy,also_view"
COLS_MAG2="title,abstract,author___affiliated_with___institution,paper___has_topic___field_of_study,paper___cites___paper"
COLS="name,details.alias,details.category,details.description,details.summary,details.summation,details.indication,details.mechanism_of_action,details.pharmacodynamics,details.mondo_definition,details.umls_description,details.orphanet_definition,details.orphanet_clinical_description,details.mayo_symptoms,details.mayo_causes,carrier,target,transporter,enzyme,expression absent,expression present,phenotype absent,phenotype present,off-label use,indication,contraindication,side effect,synergistic interaction,parent-child,linked to,associated with,interacts with,ppi"

if has 1; then
  say "1a. budgets with b = 0"
  python budgets46.py $(pairs full dbp bge) $(pairs mini mini mini) $(pairs monot5 monot5 monot5) --out b46_main > /dev/null || say "  FAILED main"
  python budgets46.py $(pairs bgedense bgedense bgedense) $(pairs bgehybrid bgehybrid bgehybrid) --out b46_firststage > /dev/null || say "  FAILED first stage"
  python budgets46.py $(for d in $QW; do echo $d:qwen8b; done) --out b46_qwen > /dev/null || say "  FAILED qwen"
  python budgets46.py $(pairs bge43 bge43 bge43) $(pairs mini43 mini43 mini43) --metric ndcg --out b46_ndcg > /dev/null || say "  FAILED ndcg"
  say "1b. calibration (design / certification / test), per pair"
  python calib46.py $(pairs full dbp bge) --out c46_bge > /dev/null || say "  FAILED"
  python calib46.py $(pairs mini mini mini) --out c46_mini > /dev/null || say "  FAILED"
  python calib46.py $(pairs monot5 monot5 monot5) --out c46_monot5 > /dev/null || say "  FAILED"
  python calib46.py $(for d in $QW; do echo $d:qwen8b; done) --ncal 100 200 --out c46_qwen > /dev/null || say "  FAILED"
  say "1c. equal total label budget (WIG / proxy choice / router / transfer router, each + certification)"
  for t in "full dbp" "mini mini" "monot5 monot5"; do set -- $t
    python calib46.py $(STK $1 $2) --mode labels --ncal 200 500 --out l46_stark_$1 > /dev/null || say "  FAILED stark $1"; done
  for t in bge mini monot5; do python calib46.py $(BR $t) --mode labels --ncal 200 --out l46_beir_$t > /dev/null || say "  FAILED beir $t"; done
  say "1d. group decomposition"
  # newer per-query files (bge43 / mini43) carry the query ids needed to look up the query text
  for t in bge43 mini43 monot5; do python groups46.py $(STK $t $t) --out g46_$t > /dev/null || say "  FAILED $t"; done
  python groups46.py $(for d in $QW; do echo $d:qwen8b; done) --out g46_qwen > /dev/null || say "  FAILED qwen"
  say "1e. re-rankers on the Qwen query ids"
  python paired46.py $QW --qwen qwen8b --others mini43 bge43 monot5 --out p46 > /dev/null || say "  FAILED"
  tar czf review46_step1.tgz b46_*.md b46_*.csv c46_*.csv c46_*_skipped.txt l46_*.csv l46_*_skipped.txt g46_*.md g46_*.csv p46.md p46.csv
  say "STEP 1 DONE: results: review46_step1.tgz"
fi

if has 2; then
  RR="cross-encoder/ms-marco-MiniLM-L-6-v2 BAAI/bge-reranker-v2-m3"
  tm() { say "2. timing $1"; python timing46.py "$1" --columns "$2" --rerankers $RR --batch 64 32 --labels-tag bge43 --out "t46_$(basename $1).json" > "t46_$(basename $1).log" 2>&1 && say "  done" || say "  FAILED (see t46_$(basename $1).log)"; }
  tm dbpedia_test    "title,abstract"
  tm mag_testfull    "$COLS_MAG2"
  tm prime_testfull  "$COLS"
  tm amazon_testfull "$COLS_AMZ"
  tar czf review46_step2.tgz t46_*.json t46_*.log
  say "STEP 2 DONE: results: review46_step2.tgz"
fi
