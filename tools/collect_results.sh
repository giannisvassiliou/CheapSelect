#!/bin/bash
# collect_results.sh: copy the per-query result files and run logs from a working directory into this repository.
#   bash tools/collect_results.sh /path/to/workdir
# Copies every <collection>/selective_pq_<tag>.jsonl to results/per_query/<collection>/, the progress logs of the run
# scripts to results/logs/, the analysis outputs (*.md, *.csv of the 46/47/48 analyses) to results/tables/, and writes
# results/MANIFEST.md5. Corpora, embeddings and model files are NOT copied (they are re-created from public sources).
set -e
SRC="${1:?usage: bash tools/collect_results.sh /path/to/workdir}"
DST="$(cd "$(dirname "$0")/.." && pwd)/results"
mkdir -p "$DST/per_query" "$DST/logs" "$DST/tables"
n=0
for f in "$SRC"/*/selective_pq_*.jsonl; do
  [ -e "$f" ] || continue
  c=$(basename "$(dirname "$f")")
  [ "$c" = "old_dupqrels" ] && continue
  mkdir -p "$DST/per_query/$c"; cp "$f" "$DST/per_query/$c/"; n=$((n+1))
done
cp "$SRC"/run_*_progress.log "$DST/logs/" 2>/dev/null || true
for p in b46 c46 l46 g46 p46 t46 h47 b47 e47 h48 b48 why42 r41 s40 s42 n43 rel43 cas43 reg43 g45 llm44; do
  cp "$SRC"/${p}*.md "$SRC"/${p}*.csv "$SRC"/${p}*.json "$DST/tables/" 2>/dev/null || true
done
(cd "$DST" && find per_query tables logs -type f | sort | xargs md5sum > MANIFEST.md5)
echo "copied $n per-query files; manifest: $DST/MANIFEST.md5 ($(wc -l < "$DST/MANIFEST.md5") files)"
du -sh "$DST"
