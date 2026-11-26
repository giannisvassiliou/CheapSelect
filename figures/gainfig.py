"""Point-estimate budget of WIG vs the re-ranker's mean gain, 27 collection-re-ranker pairs, with random's expected budget."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = json.load(open("budgets_27pairs.json"))
B = {tuple(k.split("|")): v for k, v in d["B"].items()}; G = {tuple(k.split("|")): v for k, v in d["G"].items()}
INK, MUTED = "#2b2b29", "#6b6b66"
mk = {"mini": "o", "bge": "s", "monot5": "^", "qwen8b": "D"}
grp = lambda ds: "#2a78d6" if ds.startswith("beir") is False else "#eb6834"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": "#c9c8c2", "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK})
fig, ax = plt.subplots(figsize=(3.4, 2.7))
g = np.linspace(0.05, 26, 400); grid = np.arange(0, 101, 10)
rb = [min([b for b in grid if all((1 - bb / 100) * x <= 1 + 1e-9 for bb in grid if bb >= b)]) for x in g]
ax.plot(g, rb, color="#8a8a85", ls="--", lw=1.1, zorder=1)
for (t, ds), x in G.items():
    y = B[(t, ds, "WIG-like", "1.0")][1]
    jit = {"mini": -0.6, "bge": 0, "monot5": 0.6}[t]
    ax.scatter(x, y + jit * 2, marker=mk[t], s=30, color=grp(ds), edgecolor="white", linewidth=0.7, zorder=3)
# Qwen3-Reranker-8B (sampled queries): (collection, mean gain, WIG point budget)
for ds, x, y in [("mag", 10.4, 90), ("prime", 10.8, 80), ("dbpedia", 22.9, 90), ("amazon", 9.8, 60),
                 ("beir_scifact", 10.7, 90), ("beir_nfcorpus", 7.3, 80), ("beir_scidocs", 10.3, 100)]:
    ax.scatter(x, y + 1.2, marker="D", s=24, color=grp(ds), edgecolor="white", linewidth=0.7, zorder=3)
ax.set_xlabel("Mean re-ranker gain over BM25 (MRR points)"); ax.set_ylabel("WIG budget, point estimate (%)")
ax.set_ylim(-4, 108); ax.set_xlim(-1.5, 26.5)
ax.grid(axis="y", color="#ecebe6", lw=0.8); ax.spines[["top", "right"]].set_visible(False)
h = [plt.Line2D([], [], color=MUTED, marker=mk[r], ls="", markersize=5) for r in mk]
h += [plt.Line2D([], [], color="#2a78d6", marker="o", ls=""), plt.Line2D([], [], color="#eb6834", marker="o", ls=""), plt.Line2D([], [], color="#8a8a85", ls="--")]
ax.legend(h, ["MiniLM-L6", "bge-v2-m3", "monoT5-3B", "Qwen3-8B", "STaRK / DBpedia", "BEIR", "random"], frameon=False, fontsize=6.3, loc="lower right", ncol=2)
fig.tight_layout(); fig.savefig("gain_budget.pdf", bbox_inches="tight"); fig.savefig("gain_budget.png", dpi=200, bbox_inches="tight")
