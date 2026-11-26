#!/bin/bash
# run_extra41.sh: pre-retrieval predictors + trained routers + random's real budget. CPU only (safe while
# monoT5 uses the GPU). Needs preret41.py and router41.py in the working directory.
#   tmux new -s extra 'bash run_extra41.sh'        check later:  cat run_extra41_progress.log
cd "${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}" || exit 1
[ -n "$VENV" ] && source "$VENV/bin/activate"
LOG="run_extra41_progress.log"
say() { echo "$(date '+%F %T')  $*" | tee -a "$LOG"; }
W="${WORKERS:-8}"

COLS_AMZ="title,brand,description,feature,review,qa,also_buy,also_view"
COLS_MAG2="title,abstract,author___affiliated_with___institution,paper___has_topic___field_of_study,paper___cites___paper"
COLS="name,details.alias,details.category,details.description,details.summary,details.summation,details.indication,details.mechanism_of_action,details.pharmacodynamics,details.mondo_definition,details.umls_description,details.orphanet_definition,details.orphanet_clinical_description,details.mayo_symptoms,details.mayo_causes,carrier,target,transporter,enzyme,expression absent,expression present,phenotype absent,phenotype present,off-label use,indication,contraindication,side effect,synergistic interaction,parent-child,linked to,associated with,interacts with,ppi"

# ---- 1. pre-retrieval predictors (once per collection; skipped if already there)
pre() {
  [ -s "$1/preret.jsonl" ] && { say "preret $1: exists"; return; }
  say "preret $1"
  python preret41.py "$1" --columns "$2" --workers "$W" > "preret_$(basename $1).log" 2>&1 \
    && say "  done" || say "  FAILED (see preret_$(basename $1).log)"
}
pre prime_testfull  "$COLS"
pre mag_testfull    "$COLS_MAG2"
pre amazon_testfull "$COLS_AMZ"
pre dbpedia_test    "title,abstract"
for d in scifact nfcorpus fiqa arguana scidocs; do
  [ -s "beir_$d/corpus.jsonl" ] && pre "beir_$d" "title,abstract"
done

# ---- 2. routers. STaRK + DBpedia together (so "other collections" = the other 3)
say "router STaRK+DBpedia, bge"
python router41.py amazon_testfull:full mag_testfull:full prime_testfull:full dbpedia_test:dbp --out r41_stark_bge.md > /dev/null 2>&1 || say "  FAILED"
for t in mini monot5; do
  say "router STaRK+DBpedia, $t"
  python router41.py amazon_testfull:$t mag_testfull:$t prime_testfull:$t dbpedia_test:$t --out r41_stark_$t.md > /dev/null 2>&1 || say "  FAILED"
done
# BEIR: only collections whose run for that reranker is finished
for t in bge mini monot5; do
  args=""
  for d in scifact nfcorpus fiqa arguana scidocs; do
    [ -s "beir_$d/selective_pq_$t.jsonl" ] && args="$args beir_$d:$t"
  done
  [ -n "$args" ] || continue
  say "router BEIR, $t:$args"
  python router41.py $args --out r41_beir_$t.md > /dev/null 2>&1 || say "  FAILED"
done

cat r41_*_summary.csv | awk 'NR==1 || !/^dataset/' > all_router41_summary.csv
cat r41_*_budgets.csv | awk 'NR==1 || !/^dataset/' > all_router41_budgets.csv
say "ALL DONE: results: r41_*.md and all_router41_summary.csv"
