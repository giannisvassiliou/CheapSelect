#!/bin/bash
# run_beir_all.sh: frozen selective re-ranking pipeline on small BEIR datasets, with two re-rankers, then stats.
#   tmux new -s beir      then:  bash run_beir_all.sh      then:  Ctrl+b, d
#   check later:  cat run_beir_progress.log
#   other datasets / re-rankers:
#     DATASETS="scifact nfcorpus" RERANKERS="bge" bash run_beir_all.sh
#   other GPU:  GPU=3 bash run_beir_all.sh
cd "${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}" || exit 1
[ -n "$VENV" ] && source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="${GPU:-7}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=run_beir_progress.log
say() { echo "$(date '+%F %T')  $*" | tee -a "$LOG"; }

DATASETS="${DATASETS:-scifact nfcorpus fiqa arguana scidocs}"
RERANKERS="${RERANKERS:-bge mini}"
declare -A RRNAME=( [bge]="BAAI/bge-reranker-v2-m3" [mini]="cross-encoder/ms-marco-MiniLM-L-6-v2"
                    [monot5]="castorini/monot5-3b-msmarco-10k" )
declare -A RRBS=( [bge]=32 [mini]=64 [monot5]=4 )

for f in combo_search37.py prepare_beir.py sel_stats.py qpp_gain_corr.py; do
  [ -f "$f" ] || { say "MISSING $f in the working directory - copy it first"; exit 1; }
done
say "START datasets: $DATASETS | re-rankers: $RERANKERS"

# ---- 1. data
for d in $DATASETS; do
  out="beir_${d}"
  if [ ! -s "$out/corpus.jsonl" ]; then
    say "prepare $d"
    python prepare_beir.py --name "$d" --split test --out "$out" >> "$LOG" 2>&1
    if [ ! -s "$out/corpus.jsonl" ]; then
      say "  Hugging Face failed for $d; trying the BEIR zip"
      wget -q "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/${d}.zip" \
        && unzip -q -o "${d}.zip" && python prepare_beir.py --name "$d" --beir-dir "$d" --split test --out "$out" >> "$LOG" 2>&1
    fi
  fi
  [ -s "$out/corpus.jsonl" ] && say "  $d: $(wc -l < $out/corpus.jsonl) rows, $(wc -l < $out/queries.jsonl) queries" \
                             || say "  FAILED to prepare $d"
done

# ---- 2. runs
for d in $DATASETS; do
  out="beir_${d}"
  [ -s "$out/corpus.jsonl" ] || continue
  # use only fields that exist (some BEIR sets have no titles)
  cols=$(python - "$out" <<'EOF'
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
)
  for r in $RERANKERS; do
    say "run $d with $r (columns: $cols)"
    python -u combo_search37.py run --data "$out" --columns "$cols" --gl-dataset dbpedia \
      --device cuda --batch-size 256 --no-col-filter --gliner --gl-ceiling \
      --cross-encoder "${RRNAME[$r]}" --ce-fp16 --ce-batch-size "${RRBS[$r]}" \
      --selective --timing --tag "$r" > "beir_${d}_${r}.log" 2>&1 \
      && say "  done" || say "  FAILED (see beir_${d}_${r}.log)"
  done
done

# ---- 3. stats per re-ranker over all BEIR datasets
for r in $RERANKERS; do
  dirs=""; pairs=""
  for d in $DATASETS; do
    [ -s "beir_${d}/selective_pq_${r}.jsonl" ] && dirs="$dirs beir_${d}" && pairs="$pairs beir_${d}:${r}"
  done
  [ -n "$dirs" ] || continue
  say "stats $r:$dirs"
  python sel_stats.py $dirs --tag "$r" --out "sel_stats_beir_${r}.md" >> "$LOG" 2>&1
  python qpp_gain_corr.py $pairs >> "$LOG" 2>&1 && mv qpp_gain_corr.md "qpp_gain_corr_beir_${r}.md"
done
say "ALL DONE: sel_stats_beir_<rr>.md and qpp_gain_corr_beir_<rr>.md for: $RERANKERS"
