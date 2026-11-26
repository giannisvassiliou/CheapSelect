#!/bin/bash
# run_stats40.sh: recompute all statistics with sel_stats40.py (adds point-estimate budgets + a *_budgets.csv).
# No GPU, a few minutes. Run in the working directory:   bash run_stats40.sh
cd "${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}" || exit 1
[ -n "$VENV" ] && source "$VENV/bin/activate"
B="beir_scifact beir_nfcorpus beir_fiqa beir_arguana beir_scidocs"
S="amazon_testfull mag_testfull prime_testfull"
for t in full mini monot5; do
  [ "$t" = full ] && d=dbp || d=$t
  python sel_stats40.py $S --tag $t --out s40_stark_$t.md > /dev/null
  python sel_stats40.py dbpedia_test --tag $d --out s40_dbp_$t.md > /dev/null
done
for t in bge mini monot5; do
  ds=""
  for d in $B; do [ -s "$d/selective_pq_$t.jsonl" ] && ds="$ds $d"; done   # only finished collections
  [ -n "$ds" ] && python sel_stats40.py $ds --tag $t --out s40_beir_$t.md > /dev/null
done
cat s40_*_budgets.csv | awk 'NR==1 || !/^dataset/' > all_budgets40.csv
echo "done: all_budgets40.csv ($(wc -l < all_budgets40.csv) lines)"
