"""3x4 grid of budget curves: rows = re-rankers, columns = collections (from sel_stats40 *_curves.csv)."""
import csv
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = [("WIG-like", "WIG", "#2a78d6", "o"), ("NQC-like spread", "NQC", "#eb6834", "s"),
          ("tag alarm", "tag alarm", "#1baf7a", "^"), ("BM25 margin", "BM25 margin", "#eda100", "D")]
GRAY, INK, MUTED = "#8a8a85", "#2b2b29", "#6b6b66"
ROWS = [("MiniLM-L6", "mini", "mini"), ("bge-reranker-v2-m3", "full", "full"), ("monoT5-3B", "monot5", "monot5")]
COLS = [("amazon_testfull", "STaRK-Amazon"), ("mag_testfull", "STaRK-MAG"), ("prime_testfull", "STaRK-Prime"), ("dbpedia_test", "DBpedia-Entity")]
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": "#c9c8c2", "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK})
fig, axes = plt.subplots(3, 4, figsize=(7.0, 5.0), sharex=True)
for i, (rname, st, db) in enumerate(ROWS):
    data, ref = defaultdict(lambda: defaultdict(list)), {}
    for fn in (f"s40_stark_{st}_curves.csv", f"s40_dbp_{db}_curves.csv"):
        for r in csv.DictReader(open(fn)):
            data[r["dataset"]][r["selector"]].append((int(r["budget"]), float(r["MRR"])))
            ref[r["dataset"]] = (float(r["bm25_MRR"]), float(r["all_MRR"]))
    for j, (ds, nice) in enumerate(COLS):
        ax = axes[i, j]
        bm, al = ref[ds]
        xs = [0] + [b for b, _ in sorted(data[ds]["WIG-like"])]
        ax.axhline(al, color=GRAY, lw=1.0); ax.axhline(bm, color=GRAY, lw=0.8, ls=":")
        for key, ls in (("random (expected)", "--"), ("ORACLE", ":")):
            pts = sorted(data[ds][key]); ax.plot([0] + [p[0] for p in pts], [bm] + [p[1] for p in pts], color=GRAY, lw=1.0, ls=ls)
        for key, lab, c, m in SERIES:
            pts = sorted(data[ds][key])
            ax.plot([0] + [p[0] for p in pts], [bm] + [p[1] for p in pts], color=c, lw=1.3, marker=m, ms=2.6, label=lab)
        lo = min(bm, min(v for p in data[ds].values() for _, v in p if True)) ; hi = max(v for p in data[ds].values() for _, v in p)
        ax.set_ylim(lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo))
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#ecebe6", lw=0.6)
        if i == 0: ax.set_title(nice, fontsize=8, color=INK)
        if j == 0: ax.set_ylabel(f"{rname}\nMRR (%)")
        if i == 2: ax.set_xlabel("queries re-ranked (%)")
        ax.set_xticks([0, 50, 100])
h, l = axes[0, 0].get_legend_handles_labels()
h += [plt.Line2D([], [], color=GRAY, ls="--"), plt.Line2D([], [], color=GRAY, ls=":"), plt.Line2D([], [], color=GRAY)]
l += ["random", "oracle", "re-rank all"]
fig.legend(h, l, loc="lower center", ncol=7, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig("curves_grid.pdf", bbox_inches="tight"); fig.savefig("curves_grid.png", dpi=170, bbox_inches="tight")
