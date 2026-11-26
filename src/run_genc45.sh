#!/bin/bash
# run_genc45.sh: Genc et al.'s router re-implemented on our data (GPU ~1-2 h; bert-base-uncased, ~0.4 GB download)
#   tmux new -s genc 'GPU=6 bash run_genc45.sh'        -> results: g45.md and g45.csv
cd "${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}" || exit 1
[ -n "$VENV" ] && source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="${GPU:-6}"
export PYTHONUNBUFFERED=1
for f in genc45.py lat45.py; do [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done
python lat45.py
python genc45.py --lat lat45.json --out g45 2>&1 | tee g45.log
echo "DONE: results: g45.md and g45.csv"
