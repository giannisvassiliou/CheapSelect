#!/bin/bash
# run_review42.sh: experiments for the review (points A, B, C).
#   1. conservative budget calibration, all 27 pairs            (CPU, ~10 min)        -> c42_*.md / .csv
#   2. measured end-to-end latency and throughput               (GPU, ~2-3 h)         -> lat42_*.json / .log
#   3. dense and hybrid first stage with bge, 9 collections     (GPU, ~3-4 h)         -> selective_pq_bgedense / _bgehybrid
#   4. statistics for step 3                                    (CPU, ~20 min)        -> s42_*, r42_*, c42_fs_*
# Needs in the working directory: combo_search42.py calib42.py latency42.py sel_stats40.py router41.py qpp_gain_corr.py
#   tmux new -s review 'bash run_review42.sh'          progress:  cat run_review42_progress.log
#   only some steps:  STEPS="1 4" bash run_review42.sh
cd "${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}" || exit 1
[ -n "$VENV" ] && source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="${GPU:-7}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STEPS="${STEPS:-1 2 3 4}"
LOG="run_review42_progress.log"
say() { echo "$(date '+%F %T')  $*" | tee -a "$LOG"; }
for f in combo_search42.py calib42.py latency42.py sel_stats40.py router41.py qpp_gain_corr.py; do
  [ -f "$f" ] || { say "MISSING $f - copy it first"; exit 1; }
done
python -c "import sklearn" 2>/dev/null || say "note: sklearn missing in this python - routers run ridge only"

COLS_AMZ="title,brand,description,feature,review,qa,also_buy,also_view"
COLS_MAG2="title,abstract,author___affiliated_with___institution,paper___has_topic___field_of_study,paper___cites___paper"
COLS="name,details.alias,details.category,details.description,details.summary,details.summation,details.indication,details.mechanism_of_action,details.pharmacodynamics,details.mondo_definition,details.umls_description,details.orphanet_definition,details.orphanet_clinical_description,details.mayo_symptoms,details.mayo_causes,carrier,target,transporter,enzyme,expression absent,expression present,phenotype absent,phenotype present,off-label use,indication,contraindication,side effect,synergistic interaction,parent-child,linked to,associated with,interacts with,ppi"
BEIR="scifact nfcorpus fiqa arguana scidocs"
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
has() { [[ " $STEPS " == *" $1 "* ]]; }

# ---- 1. calibration (point rule vs conservative rules), all 27 pairs
if has 1; then
  say "1. calibration"
  python calib42.py amazon_testfull:full mag_testfull:full prime_testfull:full dbpedia_test:dbp --out c42_stark_bge.md > /dev/null 2>&1 || say "  FAILED bge"
  for t in mini monot5; do
    python calib42.py amazon_testfull:$t mag_testfull:$t prime_testfull:$t dbpedia_test:$t --out c42_stark_$t.md > /dev/null 2>&1 || say "  FAILED $t"
  done
  for t in bge mini monot5; do
    python calib42.py $(for d in $BEIR; do echo beir_$d:$t; done) --out c42_beir_$t.md > /dev/null 2>&1 || say "  FAILED beir $t"
  done
  say "  done"
fi

# ---- 2. measured latency / throughput
if has 2; then
  lat() {  # data cols reranker short n cebs
    say "2. latency $1 $4"
    python latency42.py "$1" --columns "$2" --reranker "$3" --n "$5" --ce-batch-size "$6" --out "lat42_$(basename $1)_$4.json" \
      > "lat42_$(basename $1)_$4.log" 2>&1 && say "  done" || say "  FAILED (see lat42_$(basename $1)_$4.log)"
  }
  lat mag_testfull    "$COLS_MAG2"     BAAI/bge-reranker-v2-m3 bge 300 32
  lat prime_testfull  "$COLS"          BAAI/bge-reranker-v2-m3 bge 300 32
  lat amazon_testfull "$COLS_AMZ"      BAAI/bge-reranker-v2-m3 bge 300 32
  lat dbpedia_test    "title,abstract" BAAI/bge-reranker-v2-m3 bge 300 32
  lat mag_testfull    "$COLS_MAG2"     cross-encoder/ms-marco-MiniLM-L-6-v2 mini 300 64
  lat prime_testfull  "$COLS"          cross-encoder/ms-marco-MiniLM-L-6-v2 mini 300 64
  lat mag_testfull    "$COLS_MAG2"     castorini/monot5-3b-msmarco-10k monot5 100 4
  lat prime_testfull  "$COLS"          castorini/monot5-3b-msmarco-10k monot5 100 4
fi

# ---- 3. dense and hybrid first stage (bge), 9 collections
OPTS="--device cuda --batch-size 256 --no-col-filter --gliner --gl-ceiling --cross-encoder BAAI/bge-reranker-v2-m3 --ce-fp16 --ce-batch-size 32 --selective --ce-selective-only"
if has 3; then
  for fs in dense hybrid; do
    run() {  # data cols logname [extra]
      say "3. $fs $1"
      python -u combo_search42.py run --data "$1" $OPTS --first-stage $fs --tag bge$fs --columns "$2" $4 > "fs42_${fs}_$3.log" 2>&1 \
        && say "  done" || say "  FAILED (see fs42_${fs}_$3.log)"
    }
    run prime_testfull  "$COLS"          prime
    run mag_testfull    "$COLS_MAG2"     mag
    run amazon_testfull "$COLS_AMZ"      amazon
    run dbpedia_test    "title,abstract" dbp
    for d in $BEIR; do
      [ -s "beir_$d/corpus.jsonl" ] && run "beir_$d" "$(beircols beir_$d)" "beir_$d" "--gl-dataset dbpedia"
    done
  done
fi

# ---- 4. statistics for the new first stages
if has 4; then
  for fs in dense hybrid; do
    t=bge$fs
    say "4. stats $t"
    S=""; B=""
    for d in amazon_testfull mag_testfull prime_testfull dbpedia_test; do [ -s "$d/selective_pq_$t.jsonl" ] && S="$S $d"; done
    for d in $BEIR; do [ -s "beir_$d/selective_pq_$t.jsonl" ] && B="$B beir_$d"; done
    [ -n "$S" ] && python sel_stats40.py $S --tag $t --out s42_stark_$t.md > /dev/null 2>&1
    [ -n "$B" ] && python sel_stats40.py $B --tag $t --out s42_beir_$t.md > /dev/null 2>&1
    [ -n "$S" ] && python qpp_gain_corr.py $(for d in $S; do echo $d:$t; done) > qpp42_stark_$t.md 2>&1
    [ -n "$B" ] && python qpp_gain_corr.py $(for d in $B; do echo $d:$t; done) > qpp42_beir_$t.md 2>&1
    [ -n "$S" ] && python router41.py $(for d in $S; do echo $d:$t; done) --out r42_stark_$t.md > /dev/null 2>&1
    [ -n "$B" ] && python router41.py $(for d in $B; do echo $d:$t; done) --out r42_beir_$t.md > /dev/null 2>&1
  done
  cat s42_*_budgets.csv 2>/dev/null | awk 'NR==1 || !/^dataset/' > all_budgets42_fs.csv
  say "  done"
fi
tar czf review42_results.tgz c42_*.md c42_*.csv lat42_*.json lat42_*.log s42_*.md s42_*_budgets.csv s42_*_curves.csv qpp42_*.md r42_*.md r42_*_summary.csv all_budgets42_fs.csv 2>/dev/null
say "ALL DONE: results: review42_results.tgz"
