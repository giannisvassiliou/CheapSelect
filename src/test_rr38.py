#!/usr/bin/env python3
"""test_rr.py: 1-minute check that a re-ranker loads, fits in GPU memory and scores sensibly.
  CUDA_VISIBLE_DEVICES=7 python test_rr.py castorini/monot5-3b-msmarco-10k
Prints the three scores (the first, relevant document must be highest), speed and peak memory."""
import sys
import time

import torch
import transformers

from combo_search38 import make_reranker

name = sys.argv[1]
bs = int(sys.argv[2]) if len(sys.argv) > 2 else 16
print("transformers", transformers.__version__)
rr, kind = make_reranker(name, device="cuda", fp16=True)
print("backend:", kind)
q = "airsoft magazines from Elite Force"
docs = ["title: Elite Force M4 Airsoft Magazine, 140 rounds. brand: Elite Force",
        "title: Climbing rope 10mm, 60m. brand: Edelrid",
        "title: Airsoft goggles anti-fog. brand: Valken"]
print("SCORES", [round(float(x), 3) for x in rr.predict([(q, d) for d in docs])])
t = time.time()
rr.predict([(q, docs[0] * 20)] * 100, batch_size=bs)
print(f"100 long pairs (batch {bs}): {time.time() - t:.2f}s")
print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
