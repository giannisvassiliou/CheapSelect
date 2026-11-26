#!/bin/bash
# run_llm44.sh: LLM re-ranker (Qwen3-Reranker-8B) on a sample of queries, with the v43/v44 per-query dump
# (nDCG@10 + re-ranker features), then all statistics.  GPU ~10 h.  The model (~16 GB) goes to /tmp, not the working disk.
#   tmux new -s llm 'GPU=0 bash run_llm44.sh'          progress: cat run_llm44_progress.log
#   only the test:  TEST_ONLY=1 GPU=0 bash run_llm44.sh
cd "${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}" || exit 1
[ -n "$VENV" ] && source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG="run_llm44_progress.log"
say() { echo "$(date '+%F %T')  $*" | tee -a "$LOG"; }
for f in combo_search44.py sel_stats40.py router41.py why42.py calib43.py qpp_gain_corr.py ndcg43.py; do
  [ -f "$f" ] || { say "MISSING $f - copy it first"; exit 1; }
done
M=${HF_LOCAL:-/tmp/hf_models}/Qwen3-Reranker-8B
TAG=qwen8b
CEBS="${CEBS:-8}"

# ---- 0. download the model to /tmp (once)
if [ ! -s "$M/config.json" ]; then
  say "downloading Qwen/Qwen3-Reranker-8B to $M"
  mkdir -p "$M"
  python - "$M" <<'PY' >> "$LOG" 2>&1 || { say "DOWNLOAD FAILED"; exit 1; }
import sys
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-Reranker-8B", local_dir=sys.argv[1])
PY
  say "  done ($(du -sh $M | cut -f1))"
fi

# ---- 1. quick test: loads, fits, scores the relevant document highest, speed
say "test"
python - "$M" "$CEBS" <<'PY' > llm44_test.log 2>&1
import sys, time, torch
from combo_search44 import make_reranker
rr, kind = make_reranker(sys.argv[1], device="cuda", fp16=True)
bs = int(sys.argv[2])
q = "airsoft magazines from Elite Force"
docs = ["title: Elite Force M4 Airsoft Magazine, 140 rounds. brand: Elite Force",
        "title: Climbing rope 10mm, 60m. brand: Edelrid", "title: Airsoft goggles anti-fog. brand: Valken"]
sc = rr.predict([(q, d) for d in docs], batch_size=bs)
print("backend", kind, "SCORES", [round(float(x), 3) for x in sc])
t = time.time(); rr.predict([(q, docs[0] * 40)] * 100, batch_size=bs); torch.cuda.synchronize()
print(f"100 long pairs (batch {bs}): {time.time() - t:.1f}s   peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
assert sc[0] > sc[1] and sc[0] > sc[2], "relevant document not ranked first"
PY
st=$?; tee -a "$LOG" < llm44_test.log
[ $st -eq 0 ] || { say "TEST FAILED (see llm44_test.log)"; exit 1; }
[ -n "$TEST_ONLY" ] && exit 0

# ---- 2. runs on a sample (seeded; --max-queries draws a fixed random subset)
COLS_AMZ="title,brand,description,feature,review,qa,also_buy,also_view"
COLS_MAG2="title,abstract,author___affiliated_with___institution,paper___has_topic___field_of_study,paper___cites___paper"
COLS="name,details.alias,details.category,details.description,details.summary,details.summation,details.indication,details.mechanism_of_action,details.pharmacodynamics,details.mondo_definition,details.umls_description,details.orphanet_definition,details.orphanet_clinical_description,details.mayo_symptoms,details.mayo_causes,carrier,target,transporter,enzyme,expression absent,expression present,phenotype absent,phenotype present,off-label use,indication,contraindication,side effect,synergistic interaction,parent-child,linked to,associated with,interacts with,ppi"
OPTS="--device cuda --batch-size 256 --no-col-filter --gliner --gl-ceiling --cross-encoder $M --ce-fp16 --ce-batch-size $CEBS --selective --ce-selective-only --timing --tag $TAG"
run() {  # data cols nq logname [extra]
  say "run $1 ($3 queries)"
  python -u combo_search44.py run --data "$1" $OPTS --columns "$2" --max-queries "$3" $5 > "llm44_$4.log" 2>&1 \
    && say "  done" || say "  FAILED (see llm44_$4.log)"
}
run mag_testfull    "$COLS_MAG2"     1000 mag
run prime_testfull  "$COLS"          1000 prime
run dbpedia_test    "title,abstract" 0    dbp
run beir_scifact    "title,abstract" 0    scifact  "--gl-dataset dbpedia"
run beir_nfcorpus   "title,abstract" 0    nfcorpus "--gl-dataset dbpedia"
run beir_scidocs    "title,abstract" 0    scidocs  "--gl-dataset dbpedia"
run amazon_testfull "$COLS_AMZ"      500  amazon

# ---- 3. statistics
say "stats"
S=""; for d in mag_testfull prime_testfull dbpedia_test beir_scifact beir_nfcorpus beir_scidocs amazon_testfull; do
  [ -s "$d/selective_pq_$TAG.jsonl" ] && S="$S $d"; done
P=$(for d in $S; do echo $d:$TAG; done)
python sel_stats40.py $S --tag $TAG --out llm44_stats.md > /dev/null 2>&1
python qpp_gain_corr.py $P > llm44_qppcorr.md 2>&1
python router41.py $P --out llm44_router.md > /dev/null 2>&1
python why42.py $P --out llm44_why.md > /dev/null 2>&1
python calib43.py $P --out llm44_calib.md > /dev/null 2>&1
python ndcg43.py $P > /dev/null 2>&1
python sel_stats40.py $S --tag ${TAG}ndcg --out llm44_stats_ndcg.md > /dev/null 2>&1
grep -h -A 13 "=== TIMING" llm44_*.log > llm44_timing.txt 2>/dev/null
tar czf llm44_results.tgz llm44_*.md llm44_*.csv llm44_timing.txt llm44_test.log 2>/dev/null
say "ALL DONE: results: llm44_results.tgz   (the model in $M can be deleted afterwards: rm -rf ${HF_LOCAL:-/tmp/hf_models})"
