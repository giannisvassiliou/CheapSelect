"""Achieved enrichment of WIG vs required enrichment at a 50% budget (from sel_stats40 curves).
r = mean gain of skipped queries / mean gain of all queries (random: 1); safe iff r <= r* = delta / ((1-b) * mean gain)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = json.load(open("enrichment_27pairs.json"))
INK, MUTED = "#2b2b29", "#6b6b66"
mk = {"mini": "o", "bge": "s", "monot5": "^"}
NAME = {"amazon_testfull": "Amazon", "mag_testfull": "MAG", "prime_testfull": "Prime", "dbpedia_test": "DBpedia",
        "beir_scifact": "SciFact", "beir_nfcorpus": "NFCorpus", "beir_fiqa": "FiQA", "beir_arguana": "ArguAna", "beir_scidocs": "SciDocs"}
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": "#c9c8c2", "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK})
fig, ax = plt.subplots(figsize=(3.4, 3.0))
import numpy as np
xs = np.geomspace(0.05, 2, 200)
ax.fill_between(xs, xs, -0.3, color="#e8f1fb", zorder=0)
ax.plot(xs, xs, color="#8a8a85", lw=1, ls="--", zorder=1)
ax.axhline(1, color="#c9c8c2", lw=0.8, ls=":")
lab = {("bge", "mag_testfull"), ("bge", "amazon_testfull"), ("monot5", "mag_testfull"), ("bge", "dbpedia_test"), ("bge", "prime_testfull"), ("bge", "beir_fiqa")}
for k, v in d.items():
    t, ds = k.split("|")
    g = v["gain"]
    if g < 1.0:
        continue
    rs, r = 1.0 / (0.5 * g), v["r"]["WIG-like"]["50"]
    col = "#2a78d6" if not ds.startswith("beir") else "#eb6834"
    ax.scatter(rs, r, marker=mk[t], s=30, color=col, edgecolor="white", linewidth=0.7, zorder=3)
    if (t, ds) in lab:
        ax.annotate(f"{NAME[ds]}/{t}", (rs, r), xytext=(4, 2), textcoords="offset points", fontsize=6.3, color=INK)
ax.set_xscale("log"); ax.set_xlim(0.06, 1.6); ax.set_ylim(-0.3, 1.25)
ax.set_xlabel("required enrichment $r^*$ at 50% budget (log)"); ax.set_ylabel("WIG's enrichment $r$ at 50% budget")
ax.text(0.75, -0.22, "safe: $r\\leq r^*$", fontsize=7, color="#2a78d6")
ax.text(0.07, 1.04, "random ($r=1$)", fontsize=6.5, color=MUTED)
ax.spines[["top", "right"]].set_visible(False)
h = [plt.Line2D([], [], color=MUTED, marker=mk[r], ls="", markersize=5) for r in mk]
h += [plt.Line2D([], [], color="#2a78d6", marker="o", ls=""), plt.Line2D([], [], color="#eb6834", marker="o", ls="")]
ax.legend(h, ["MiniLM", "bge", "monoT5", "STaRK/DBpedia", "BEIR"], frameon=False, fontsize=6.3, loc="lower left", ncol=1)
fig.tight_layout(); fig.savefig("enrichment.pdf", bbox_inches="tight"); fig.savefig("enrichment.png", dpi=200, bbox_inches="tight")
