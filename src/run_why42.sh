#!/bin/bash
# run_why42.sh: enrichment / decomposition for all 27 pairs (CPU, ~1 min). Run in the working directory:  bash run_why42.sh
cd "${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}" || exit 1
[ -n "$VENV" ] && source "$VENV/bin/activate"
python why42.py amazon_testfull:full mag_testfull:full prime_testfull:full dbpedia_test:dbp --out w42_stark_bge.md > /dev/null
for t in mini monot5; do
  python why42.py amazon_testfull:$t mag_testfull:$t prime_testfull:$t dbpedia_test:$t --out w42_stark_$t.md > /dev/null
done
for t in bge mini monot5; do
  python why42.py beir_scifact:$t beir_nfcorpus:$t beir_fiqa:$t beir_arguana:$t beir_scidocs:$t --out w42_beir_$t.md > /dev/null
done
tar czf why_calib.tgz w42_*.md w42_*.csv c42_*.md c42_*.csv
echo "done: results: why_calib.tgz"
