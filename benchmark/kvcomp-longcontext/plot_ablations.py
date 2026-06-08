"""
Ablation study figures for TaperKV / HALO papers.

Generates:
  papers/taperkv_paper/figures/fig_bmin_sweep.png
  papers/taperkv_paper/figures/fig_budget_sweep.png
  papers/taperkv_paper/figures/fig_kv_asymmetry.png
  papers/halo_paper/figures/fig_halo_lever.png

Run from benchmark/kvcomp-longcontext/:
  python plot_ablations.py
"""

from __future__ import annotations
import json
import os
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

HERE   = os.path.dirname(os.path.abspath(__file__))
PAPERS = os.path.join(HERE, "..", "..", "papers")
TAPERKV_FIG = os.path.join(PAPERS, "taperkv_paper", "figures")
HALO_FIG    = os.path.join(PAPERS, "halo_paper",    "figures")
os.makedirs(TAPERKV_FIG, exist_ok=True)
os.makedirs(HALO_FIG,    exist_ok=True)

# ── palette ─────────────────────────────────────────────────────────────────
PAL = {
    "FullKV":        "#444444",
    "StreamingLLM":  "#e07040",
    "ThinKV":        "#7aaa4a",
    "TaperKV":       "#2176ae",
    "TaperKV-3bit":  "#5bc0eb",
    "KIVI-INT2":     "#9a5fbf",
    "HALO-3bit":     "#f7b731",
    "TurboQuant-3bit":"#aaaaaa",
    "TurboQuant-4bit":"#cccccc",
}
MARKER = {
    "FullKV": "D", "StreamingLLM": "x", "ThinKV": "s",
    "TaperKV": "o", "TaperKV-3bit": "^", "KIVI-INT2": "P",
    "HALO-3bit": "*", "TurboQuant-3bit": "v", "TurboQuant-4bit": "v",
}


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)["results"]


# ── helpers ──────────────────────────────────────────────────────────────────
def _set_style(ax, xlabel, ylabel, title="", ylim=None, legend=True):
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3, linestyle="--")
    if legend:
        ax.legend(fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Figure 1 – b_min sweep (TaperKV key-floor sensitivity)                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def fig_bmin_sweep():
    """Left: Llama-3.1-8B  |  Right: R1-Distill-Qwen-7B"""
    datasets = [
        (load(os.path.join(HERE, "results_ablation_bmin_llama", "results_llama-3.1-8b-instruct.json")),
         "Llama-3.1-8B-Instruct"),
        (load(os.path.join(HERE, "results_ablation_bmin_r1q7", "results_DeepSeek-R1-Distill-Qwen-7B.json")),
         "R1-Distill-Qwen-7B"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    fig.suptitle("TaperKV $b_{\\mathrm{min}}$ Key-Floor Sensitivity", fontsize=13, fontweight="bold")

    bmin_labels_3 = [2, 3, 4, 5]
    bmin_labels_4 = [2, 4, 6]

    for ax, (results, model_name) in zip(axes, datasets):
        fkv  = results["FullKV"]["needle"]["accuracy"]
        ref3 = results.get("TurboQuant-3bit", results.get("HALO-3bit", {})).get("needle", {}).get("accuracy", None)

        # 3-bit sweep
        vals3 = [results.get(f"TaperKV-3bit-bmin{b}", {}).get("needle", {}).get("accuracy", np.nan)
                 for b in bmin_labels_3]
        ax.plot(bmin_labels_3, vals3, "o-", color="#2176ae", linewidth=2,
                markersize=7, label="TaperKV 3-bit ($b_{\\min}$ sweep)", zorder=3)

        # 4-bit sweep
        vals4 = [results.get(f"TaperKV-4bit-bmin{b}", {}).get("needle", {}).get("accuracy", np.nan)
                 for b in bmin_labels_4]
        ax.plot(bmin_labels_4, vals4, "s--", color="#5bc0eb", linewidth=2,
                markersize=7, label="TaperKV 4-bit ($b_{\\min}$ sweep)", zorder=3)

        # reference lines
        ax.axhline(fkv, linestyle=":", color="#444444", linewidth=1.4, label=f"FullKV ({fkv:.2f})")
        if ref3 is not None:
            ax.axhline(ref3, linestyle="-.", color="#f7b731", linewidth=1.4, label=f"HALO-3bit ({ref3:.2f})")

        ax.axvline(3, linestyle="--", color="#e07040", alpha=0.5, linewidth=1.2)
        ax.text(3.05, ax.get_ylim()[0] + 0.02, "$b_{\\min}=3$", fontsize=7.5, color="#e07040")

        ax.set_xticks([2, 3, 4, 5, 6])
        ax.set_xlabel("$b_{\\min}$ (key bit-width floor)", fontsize=10)
        ax.set_ylabel("Needle Accuracy", fontsize=10)
        ax.set_title(model_name, fontsize=10)
        ax.set_ylim(-0.05, 1.08)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=8, framealpha=0.9, loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(TAPERKV_FIG, "fig_bmin_sweep.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved → {out}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Figure 2 – Memory-budget trade-off curve                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def fig_budget_sweep():
    """Needle accuracy vs target_mem budget, Llama-8B and R1-Q7 side-by-side."""
    budgets = [10, 15, 20, 25, 35]

    def load_budget(prefix, pcts):
        out = {}
        for p in pcts:
            pattern = os.path.join(HERE, f"results_ablation_budget_{prefix}_{p}", "results_*.json")
            files = glob.glob(pattern)
            if not files:
                continue
            r = load(files[0])
            for method, vals in r.items():
                out.setdefault(method, {})[p] = vals.get("needle", {}).get("accuracy", np.nan)
        return out

    llama_data = load_budget("llama", budgets)
    r1q7_data  = load_budget("r1q7",  budgets)

    methods_show = ["TaperKV", "TaperKV-3bit", "StreamingLLM", "ThinKV", "KIVI-INT2", "HALO-3bit"]
    line_styles  = ["-", "--", "-.", ":", "-", "--"]
    colors = ["#2176ae", "#5bc0eb", "#e07040", "#7aaa4a", "#9a5fbf", "#f7b731"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    fig.suptitle("Memory-Budget Trade-off (Needle Accuracy)", fontsize=13, fontweight="bold")

    for ax, (data, model_name) in zip(axes, [(llama_data, "Llama-3.1-8B-Instruct"),
                                              (r1q7_data,  "R1-Distill-Qwen-7B")]):
        fkv = list(data.get("FullKV", {}).values())
        if fkv:
            ax.axhline(fkv[0], linestyle=":", color="#444444", linewidth=1.4,
                       label=f"FullKV ({fkv[0]:.2f})")

        for method, ls, color in zip(methods_show, line_styles, colors):
            if method not in data:
                continue
            xs = sorted(data[method].keys())
            ys = [data[method][x] for x in xs]
            label = method.replace("TaperKV-3bit", "TaperKV-3bit").replace("KIVI-INT2", "KIVI-INT2")
            ax.plot(xs, ys, marker=MARKER.get(method, "o"), linestyle=ls,
                    color=color, linewidth=2, markersize=6, label=label, zorder=3)

        ax.set_xlabel("Target Memory Budget (%)", fontsize=10)
        ax.set_ylabel("Needle Accuracy", fontsize=10)
        ax.set_title(model_name, fontsize=10)
        ax.set_xticks(budgets)
        ax.set_xticklabels([f"{b}%" for b in budgets])
        ax.set_ylim(-0.05, 1.08)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=8, framealpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(TAPERKV_FIG, "fig_budget_sweep.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved → {out}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Figure 3 – TaperKV K/V asymmetry ablation (Ministral + R1-Q7)         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def fig_kv_asymmetry():
    """Bar chart: Needle + PPL for asym / sym / revsym + baselines."""
    datasets = [
        (load(os.path.join(HERE, "results_ablation_minis", "results_Ministral-8B-Instruct-2410.json")),
         "Ministral-8B"),
        (load(os.path.join(HERE, "results_ablation_r1q7", "results_DeepSeek-R1-Distill-Qwen-7B.json")),
         "R1-Distill-Qwen-7B"),
    ]

    kv_methods = ["TaperKV-asym", "TaperKV-sym", "TaperKV-revsym"]
    kv_labels  = ["K>V (asym)", "K=V (sym)", "V>K (rev-sym)"]
    kv_colors  = ["#2176ae", "#7aaa4a", "#e07040"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("TaperKV K/V Bit-Asymmetry Ablation", fontsize=13, fontweight="bold")

    for col, (results, model_name) in enumerate(datasets):
        ax_needle = axes[0, col]
        ax_ppl    = axes[1, col]

        fkv_needle = results["FullKV"]["needle"]["accuracy"]
        fkv_ppl    = results["FullKV"]["ppl"]["ppl"]

        needles = [results.get(m, {}).get("needle", {}).get("accuracy", 0) for m in kv_methods]
        ppls    = [results.get(m, {}).get("ppl",    {}).get("ppl",    0)  for m in kv_methods]

        x = np.arange(len(kv_methods))
        bars = ax_needle.bar(x, needles, color=kv_colors, alpha=0.85, edgecolor="white", linewidth=0.8)
        ax_needle.axhline(fkv_needle, linestyle="--", color="#444444", linewidth=1.4, label=f"FullKV ({fkv_needle:.2f})")
        ax_needle.set_xticks(x)
        ax_needle.set_xticklabels(kv_labels, fontsize=9)
        ax_needle.set_ylabel("Needle Accuracy", fontsize=10)
        ax_needle.set_title(model_name, fontsize=10)
        ax_needle.set_ylim(0, 1.15)
        ax_needle.legend(fontsize=8)
        ax_needle.grid(axis="y", alpha=0.3, linestyle="--")
        ax_needle.spines["top"].set_visible(False)
        ax_needle.spines["right"].set_visible(False)
        for bar, val in zip(bars, needles):
            ax_needle.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                           f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        bars2 = ax_ppl.bar(x, ppls, color=kv_colors, alpha=0.85, edgecolor="white", linewidth=0.8)
        ax_ppl.axhline(fkv_ppl, linestyle="--", color="#444444", linewidth=1.4, label=f"FullKV ({fkv_ppl:.2f})")
        ax_ppl.set_xticks(x)
        ax_ppl.set_xticklabels(kv_labels, fontsize=9)
        ax_ppl.set_ylabel("Perplexity ↓", fontsize=10)
        ax_ppl.legend(fontsize=8)
        ax_ppl.grid(axis="y", alpha=0.3, linestyle="--")
        ax_ppl.spines["top"].set_visible(False)
        ax_ppl.spines["right"].set_visible(False)
        for bar, val in zip(bars2, ppls):
            ax_ppl.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = os.path.join(TAPERKV_FIG, "fig_kv_asymmetry.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved → {out}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Figure 4 – HALO lever ablation (PPL iso-quality)                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def fig_halo_lever():
    """Grouped bar: HALO lever variants vs TurboQuant at 3-bit and 4-bit."""
    datasets = [
        (load(os.path.join(HERE, "results_ablation_minis", "results_Ministral-8B-Instruct-2410.json")),
         "Ministral-8B"),
        (load(os.path.join(HERE, "results_ablation_llama", "results_llama-3.1-8b-instruct.json")),
         "Llama-3.1-8B"),
    ]

    lever_methods = {
        3: ["TurboQuant-3bit", "HALO-h-noLS-3", "HALO-hg-noLS-3", "HALO-hg-LS-3", "HALO-dense-LS-3"],
        4: ["TurboQuant-4bit", "HALO-h-noLS-4", "HALO-hg-noLS-4", "HALO-hg-LS-4", "HALO-dense-LS-4"],
    }
    lever_labels = {
        3: ["TurboQuant", "HALO\nh-noLS", "HALO\nhg-noLS", "HALO\nhg-LS\n(full)", "HALO\ndense-LS"],
        4: ["TurboQuant", "HALO\nh-noLS", "HALO\nhg-noLS", "HALO\nhg-LS\n(full)", "HALO\ndense-LS"],
    }
    lever_colors = ["#aaaaaa", "#96c1f7", "#5bc0eb", "#2176ae", "#8e44ad"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle("HALO Lever Ablation — PPL (iso-quality verification)", fontsize=13, fontweight="bold")

    for row, bits in enumerate([3, 4]):
        for col, (results, model_name) in enumerate(datasets):
            ax = axes[row, col]
            fkv_ppl = results["FullKV"]["ppl"]["ppl"]
            ppls = [results.get(m, {}).get("ppl", {}).get("ppl", 0) for m in lever_methods[bits]]

            x = np.arange(len(lever_methods[bits]))
            bars = ax.bar(x, ppls, color=lever_colors, alpha=0.88, edgecolor="white", linewidth=0.8)
            ax.axhline(fkv_ppl, linestyle="--", color="#444444", linewidth=1.4,
                       label=f"FullKV PPL ({fkv_ppl:.2f})")

            ax.set_xticks(x)
            ax.set_xticklabels(lever_labels[bits], fontsize=8)
            ax.set_ylabel("Perplexity ↓", fontsize=10)
            title = f"{model_name}  ({bits}-bit)"
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ppl_min = fkv_ppl * 0.95
            ppl_max = max(ppls) * 1.06 if ppls else fkv_ppl * 1.2
            ax.set_ylim(ppl_min, ppl_max)

            for bar, val in zip(bars, ppls):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = os.path.join(HALO_FIG, "fig_halo_lever.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved → {out}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Figure 5 – HALO lever: Needle accuracy for reasoning model (R1-Q7)     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def fig_halo_lever_reasoning():
    """Additional HALO lever plot for R1-Q7 (needle + PPL)."""
    results_r1q7 = load(os.path.join(HERE, "results_ablation_r1q7",
                                      "results_DeepSeek-R1-Distill-Qwen-7B.json"))
    results_r1l8 = load(os.path.join(HERE, "results_ablation_r1l8",
                                      "results_DeepSeek-R1-Distill-Llama-8B.json"))

    lever_methods_3 = ["TurboQuant-3bit", "HALO-h-noLS-3", "HALO-hg-noLS-3", "HALO-hg-LS-3", "HALO-dense-LS-3"]
    lever_labels_3  = ["TurboQuant", "h-noLS", "hg-noLS", "hg-LS\n(full)", "dense-LS"]
    lever_colors    = ["#aaaaaa", "#96c1f7", "#5bc0eb", "#2176ae", "#8e44ad"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle("HALO Lever Ablation — Reasoning Models (3-bit)", fontsize=13, fontweight="bold")

    for col, (results, model_name, metric, ylabel, ylim) in enumerate([
        (results_r1l8, "R1-Distill-Llama-8B", "needle", "Needle Accuracy", (0, 1.1)),
        (results_r1l8, "R1-Distill-Llama-8B", "ppl",    "Perplexity ↓",    None),
        (results_r1q7, "R1-Distill-Qwen-7B",  "needle", "Needle Accuracy", (0, 1.1)),
        (results_r1q7, "R1-Distill-Qwen-7B",  "ppl",    "Perplexity ↓",    None),
    ]):
        ax = axes[col]
        fkv_val = results["FullKV"][metric]["accuracy" if metric == "needle" else "ppl"]
        vals = [results.get(m, {}).get(metric, {}).get(
                    "accuracy" if metric == "needle" else "ppl", 0)
                for m in lever_methods_3]

        x = np.arange(len(lever_methods_3))
        bars = ax.bar(x, vals, color=lever_colors, alpha=0.88, edgecolor="white", linewidth=0.8)
        ax.axhline(fkv_val, linestyle="--", color="#444444", linewidth=1.4,
                   label=f"FullKV ({fkv_val:.2f})")
        ax.set_xticks(x)
        ax.set_xticklabels(lever_labels_3, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"{model_name}\n({ylabel.split()[0]})", fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(HALO_FIG, "fig_halo_lever_reasoning.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved → {out}")


if __name__ == "__main__":
    print("Generating ablation figures …")
    fig_bmin_sweep()
    fig_budget_sweep()
    fig_kv_asymmetry()
    fig_halo_lever()
    fig_halo_lever_reasoning()
    print("Done.")
