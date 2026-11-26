#!/bin/bash
# run_dbpedia.sh: frozen tag-alarm test on DBpedia-Entity v2 (BEIR test split). Run inside tmux:
#   tmux new -s dbp     then:  bash run_dbpedia.sh     then:  Ctrl+b, d
#   check:  cat run_dbpedia_progress.log        or:  tmux attach -t dbp

cd "${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}" || exit 1
[ -n "$VENV" ] && source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="${GPU:-7}"
export PYTHONUNBUFFERED=1
LOG=run_dbpedia_progress.log
say() { echo "$(date '+%F %T')  $*" | tee -a "$LOG"; }

OPTS="--device cuda --batch-size 256 --no-col-filter --gliner --gl-ceiling --cross-encoder BAAI/bge-reranker-v2-m3 --ce-fp16 --selective --timing --tag dbp"

say "START"
if [ ! -s dbpedia_test/corpus.jsonl ]; then
  say "download + convert (Hugging Face)"
  python prepare_beir.py --split test --out dbpedia_test >> "$LOG" 2>&1
  if [ ! -s dbpedia_test/corpus.jsonl ]; then
    say "Hugging Face failed; trying the BEIR zip"
    wget -q https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/dbpedia-entity.zip \
      && unzip -q -o dbpedia-entity.zip \
      && python prepare_beir.py --beir-dir dbpedia-entity --split test --out dbpedia_test >> "$LOG" 2>&1
  fi
fi
[ -s dbpedia_test/corpus.jsonl ] || { say "FAILED: no data"; exit 1; }
say "rows: $(wc -l < dbpedia_test/corpus.jsonl)  queries: $(wc -l < dbpedia_test/queries.jsonl)"

say "run (index building on ~4.6M rows takes a while)"
python -u combo_search34.py run --data dbpedia_test $OPTS --columns "title,abstract" > dbp_test.log 2>&1 \
  && say "run done" || { say "FAILED run (see dbp_test.log)"; exit 1; }

grep "tag alarm fires on" dbp_test.log | tee -a "$LOG"
python sel_stats.py dbpedia_test --tag dbp --out sel_stats_dbp.md >> "$LOG" 2>&1 \
  && say "ALL DONE: see sel_stats_dbp.md" || say "FAILED stats"
