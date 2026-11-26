#!/bin/bash
# run_rr_all38.sh: test a re-ranker, then (only if the test passes) run all 4 collections + stats with it.
#   tmux new -s rr      then:  bash run_rr_all38.sh      then:  Ctrl+b, d
#   default: monoT5-3B.  Other model:  RR=mixedbread-ai/mxbai-rerank-large-v1 TAG=mxbai bash run_rr_all38.sh
#   other GPU:  GPU=3 bash run_rr_all38.sh
#   check later:  cat run_rr_<TAG>_progress.log      (test output: rr_<TAG>_test.log)
cd "${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}" || exit 1
[ -n "$VENV" ] && source "$VENV/bin/activate"
RR="${RR:-castorini/monot5-3b-msmarco-10k}"
TAG="${TAG:-monot5}"
export CUDA_VISIBLE_DEVICES="${GPU:-7}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CEBS="${CEBS:-4}"   # re-ranker batch size (small = less GPU memory)
LOG="run_rr_${TAG}_progress.log"
say() { echo "$(date '+%F %T')  $*" | tee -a "$LOG"; }

for f in combo_search38.py test_rr38.py sel_stats.py qpp_gain_corr.py; do
  [ -f "$f" ] || { say "MISSING $f in the working directory - copy it first"; exit 1; }
done

# ---- 1. quick test
say "TEST $RR on GPU $CUDA_VISIBLE_DEVICES"
python test_rr38.py "$RR" "$CEBS" > "rr_${TAG}_test.log" 2>&1
st=$?
tee -a "$LOG" < "rr_${TAG}_test.log"
[ $st -eq 0 ] || { say "TEST FAILED (see rr_${TAG}_test.log) - full run NOT started"; exit 1; }
python - "$TAG" <<'EOF' >> "$LOG" 2>&1
import re, sys
t = open(f"rr_{sys.argv[1]}_test.log").read()
m = re.search(r"SCORES \[([-\d.e, ]+)\]", t)
sc = [float(x) for x in m.group(1).split(",")] if m else []
if len(sc) != 3 or not (sc[0] > sc[1] and sc[0] > sc[2]):
    print("CHECK FAILED: relevant document not scored highest:", sc); sys.exit(1)
print("CHECK OK:", sc)
EOF
[ $? -eq 0 ] || { say "TEST CHECK FAILED - full run NOT started"; exit 1; }
say "test passed - starting full run with $RR (tag $TAG)"

# ---- 2. full run, all 4 collections
COLS_AMZ="title,brand,description,feature,review,qa,also_buy,also_view"
COLS_MAG2="title,abstract,author___affiliated_with___institution,paper___has_topic___field_of_study,paper___cites___paper"
COLS="name,details.alias,details.category,details.description,details.summary,details.summation,details.indication,details.mechanism_of_action,details.pharmacodynamics,details.mondo_definition,details.umls_description,details.orphanet_definition,details.orphanet_clinical_description,details.mayo_symptoms,details.mayo_causes,carrier,target,transporter,enzyme,expression absent,expression present,phenotype absent,phenotype present,off-label use,indication,contraindication,side effect,synergistic interaction,parent-child,linked to,associated with,interacts with,ppi"
OPTS="--device cuda --batch-size 256 --no-col-filter --gliner --gl-ceiling --cross-encoder $RR --ce-fp16 --selective --ce-selective-only --timing --tag $TAG --ce-batch-size $CEBS"

run() {
  say "run $1"
  python -u combo_search38.py run --data "$1" $OPTS --columns "$2" > "${TAG}_$3.log" 2>&1 && say "done $1" || say "FAILED $1 (see ${TAG}_$3.log)"
}
run prime_testfull  "$COLS"          prime
run mag_testfull    "$COLS_MAG2"     mag
run amazon_testfull "$COLS_AMZ"      amazon
run dbpedia_test    "title,abstract" dbp

# ---- 3. stats
say "stats ..."
python sel_stats.py amazon_testfull mag_testfull prime_testfull --tag "$TAG" --out "sel_stats_${TAG}.md" >> "$LOG" 2>&1
python sel_stats.py dbpedia_test --tag "$TAG" --out "sel_stats_${TAG}_dbp.md" >> "$LOG" 2>&1
python qpp_gain_corr.py amazon_testfull:"$TAG" mag_testfull:"$TAG" prime_testfull:"$TAG" dbpedia_test:"$TAG" >> "$LOG" 2>&1
mv qpp_gain_corr.md "qpp_gain_corr_${TAG}.md" 2>/dev/null
grep "tag alarm fires on" ${TAG}_*.log | tee -a "$LOG"
say "ALL DONE: sel_stats_${TAG}.md, sel_stats_${TAG}_dbp.md, qpp_gain_corr_${TAG}.md"
