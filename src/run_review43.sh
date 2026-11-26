#!/bin/bash
# run_review43.sh: experiments for the second review.
#   STEP 1 (CPU, now, ~5 min, existing files): risk-control calibration (calib43), cascade with MiniLM (cascade43),
#          label-free regime predictor (regime43)                                   -> c43_*, k43_*, g43_*
#   STEP 2 (GPU, AFTER run_review42 has finished, ~3 h): MiniLM and bge re-runs with combo_search43 (adds nDCG@10 and
#          re-ranker features to the per-query files; tags mini43 / bge43)
#   STEP 3 (CPU, after step 2, ~20 min): nDCG@10 versions of all statistics, cascade and regime with the v43
#          features, relative tolerance (2% of re-rank-all MRR)                     -> n43_*, k43v_*, g43v_*, rel43_*
#   STEPS="1" bash run_review43.sh          tmux new -s r43 'STEPS="2 3" bash run_review43.sh'
cd "${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}" || exit 1
[ -n "$VENV" ] && source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="${GPU:-7}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STEPS="${STEPS:-1}"
LOG="run_review43_progress.log"
say() { echo "$(date '+%F %T')  $*" | tee -a "$LOG"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
for f in combo_search43.py calib43.py cascade43.py regime43.py ndcg43.py sel_stats40.py router41.py why42.py; do
  [ -f "$f" ] || { say "MISSING $f - copy it first"; exit 1; }
done
BEIR="scifact nfcorpus fiqa arguana scidocs"
STARK="amazon_testfull mag_testfull prime_testfull"
COLS_AMZ="title,brand,description,feature,review,qa,also_buy,also_view"
COLS_MAG2="title,abstract,author___affiliated_with___institution,paper___has_topic___field_of_study,paper___cites___paper"
COLS="name,details.alias,details.category,details.description,details.summary,details.summation,details.indication,details.mechanism_of_action,details.pharmacodynamics,details.mondo_definition,details.umls_description,details.orphanet_definition,details.orphanet_clinical_description,details.mayo_symptoms,details.mayo_causes,carrier,target,transporter,enzyme,expression absent,expression present,phenotype absent,phenotype present,off-label use,indication,contraindication,side effect,synergistic interaction,parent-child,linked to,associated with,interacts with,ppi"
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
pairs() {  # $1 = stark tag (full|mini|monot5), $2 = dbpedia tag, $3 = beir tag
  for d in $STARK; do echo "$d:$1"; done; echo "dbpedia_test:$2"; for d in $BEIR; do echo "beir_$d:$3"; done
}
ALL27="$(pairs full dbp bge) $(pairs mini mini mini) $(pairs monot5 monot5 monot5)"

# ---------------- STEP 1: CPU, existing files
if has 1; then
  say "1. calib43 (27 pairs)"
  python calib43.py $(pairs full dbp bge) --out c43_bge.md > /dev/null 2>&1 || say "  FAILED bge"
  python calib43.py $(pairs mini mini mini) --out c43_mini.md > /dev/null 2>&1 || say "  FAILED mini"
  python calib43.py $(pairs monot5 monot5 monot5) --out c43_monot5.md > /dev/null 2>&1 || say "  FAILED monot5"
  say "1. cascade43 (MiniLM as proxy / fallback; pre-v43 features)"
  python cascade43.py amazon_testfull mag_testfull prime_testfull dbpedia_test:dbp --cheap mini --exp full \
     --cheap-ms 146 98 80 47 --exp-ms 316 257 211 87 --out k43_stark_bge.md > /dev/null 2>&1 || say "  FAILED"
  python cascade43.py amazon_testfull mag_testfull prime_testfull dbpedia_test --cheap mini --exp monot5 \
     --cheap-ms 146 98 80 47 --exp-ms 3559 3259 3160 1444 --out k43_stark_monot5.md > /dev/null 2>&1 || say "  FAILED"
  python cascade43.py $(for d in $BEIR; do echo beir_$d; done) --cheap mini --exp bge --out k43_beir_bge.md > /dev/null 2>&1 || say "  FAILED"
  python cascade43.py $(for d in $BEIR; do echo beir_$d; done) --cheap mini --exp monot5 --out k43_beir_monot5.md > /dev/null 2>&1 || say "  FAILED"
  say "1. regime43 (27 pairs, top-3 probe)"
  python regime43.py $ALL27 --out g43_all.md > /dev/null 2>&1 || say "  FAILED"
  tar czf review43_step1.tgz c43_*.md c43_*.csv k43_*.md g43_*.md
  say "STEP 1 DONE: results: review43_step1.tgz"
fi

# ---------------- STEP 2: GPU re-runs with v43 dumps
if has 2; then
  RRN=(cross-encoder/ms-marco-MiniLM-L-6-v2 BAAI/bge-reranker-v2-m3); RRT=(mini43 bge43); RRB=(64 32)
  for i in 0 1; do
    OPTS="--device cuda --batch-size 256 --no-col-filter --gliner --gl-ceiling --cross-encoder ${RRN[$i]} --ce-fp16 --ce-batch-size ${RRB[$i]} --selective --ce-selective-only --tag ${RRT[$i]}"
    run() {
      say "2. ${RRT[$i]} $1"
      python -u combo_search43.py run --data "$1" $OPTS --columns "$2" $4 > "v43_${RRT[$i]}_$3.log" 2>&1 && say "  done" || say "  FAILED (see v43_${RRT[$i]}_$3.log)"
    }
    run prime_testfull  "$COLS"          prime
    run mag_testfull    "$COLS_MAG2"     mag
    run amazon_testfull "$COLS_AMZ"      amazon
    run dbpedia_test    "title,abstract" dbp
    for d in $BEIR; do run "beir_$d" "$(beircols beir_$d)" "beir_$d" "--gl-dataset dbpedia"; done
  done
fi

# ---------------- STEP 3: CPU on v43 files
if has 3; then
  P43B="$(for d in $STARK dbpedia_test; do echo $d:bge43; done) $(for d in $BEIR; do echo beir_$d:bge43; done)"
  P43M="$(for d in $STARK dbpedia_test; do echo $d:mini43; done) $(for d in $BEIR; do echo beir_$d:mini43; done)"
  say "3. nDCG@10 versions"
  python ndcg43.py $P43B $P43M > /dev/null 2>&1 || say "  FAILED ndcg43"
  for t in bge43 mini43; do
    python sel_stats40.py $STARK dbpedia_test --tag ${t}ndcg --out n43_stark_$t.md > /dev/null 2>&1
    python sel_stats40.py $(for d in $BEIR; do echo beir_$d; done) --tag ${t}ndcg --out n43_beir_$t.md > /dev/null 2>&1
    python why42.py $(for d in $STARK dbpedia_test; do echo $d:${t}ndcg; done) $(for d in $BEIR; do echo beir_$d:${t}ndcg; done) --out n43_why_$t.md > /dev/null 2>&1
    python calib43.py $(for d in $STARK dbpedia_test; do echo $d:${t}ndcg; done) $(for d in $BEIR; do echo beir_$d:${t}ndcg; done) --out n43_calib_$t.md > /dev/null 2>&1
  done
  say "3. cascade + regime with v43 features"
  python cascade43.py amazon_testfull mag_testfull prime_testfull dbpedia_test --cheap mini43 --exp bge43 \
     --cheap-ms 146 98 80 47 --exp-ms 316 257 211 87 --out k43v_stark_bge.md > /dev/null 2>&1
  python cascade43.py $(for d in $BEIR; do echo beir_$d; done) --cheap mini43 --exp bge43 --out k43v_beir_bge.md > /dev/null 2>&1
  python cascade43.py amazon_testfull mag_testfull prime_testfull dbpedia_test --cheap mini43 --exp monot5 \
     --cheap-ms 146 98 80 47 --exp-ms 3559 3259 3160 1444 --out k43v_stark_monot5.md > /dev/null 2>&1
  python regime43.py $P43B $P43M --out g43v_all.md > /dev/null 2>&1
  say "3. relative tolerance: delta = 2% of the re-rank-all MRR, per collection"
  for p in $ALL27; do
    d=${p%%:*}; t=${p##*:}
    rel=$(python -c "import json,sys; r=[json.loads(l) for l in open('$d/selective_pq_$t.jsonl')]; print(round(2*sum(x['ce']['MRR'] for x in r)/len(r),2))")
    python sel_stats40.py $d --tag $t --ni-margins $rel --cal-sizes 100 --cal-reps 20 --out rel43_$(basename $d)_$t.md > /dev/null 2>&1
  done
  cat rel43_*_budgets.csv | awk 'NR==1 || !/^dataset/' > all_rel43_budgets.csv
  tar czf review43_step3.tgz n43_*.md n43_*.csv k43v_*.md g43v_*.md all_rel43_budgets.csv
  say "STEP 3 DONE: results: review43_step3.tgz"
fi
