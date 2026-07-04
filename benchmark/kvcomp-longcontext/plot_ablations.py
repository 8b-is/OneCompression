"""
plot_ablations.py  –  paper-quality ablation figures for TaperKV / HayateQuant
(fig_halo_lever* are legacy HALO-paper figures, kept for reproducibility)

Outputs
-------
  papers/taperkv_paper/figures/
    fig_bmin_sweep.png
    fig_budget_sweep.png
    fig_kv_asymmetry.png
    fig_kv_asymmetry_reason.png
  papers/halo_paper/figures/
    fig_halo_lever.png
    fig_halo_lever_reasoning.png

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
import numpy as np

# ── Paper style (as specified) ────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':           'sans-serif',
    'xtick.direction':       'in',
    'ytick.direction':       'in',
    'xtick.major.width':     1.0,
    'ytick.major.width':     1.0,
    'axes.linewidth':        1.0,
    'axes.xmargin':          0.01,
    'axes.ymargin':          0.01,
    'legend.fancybox':       False,
    'legend.framealpha':     1,
    'legend.edgecolor':      'black',
    'mathtext.fontset':      'stix',
})

LINE_STYLES = ['-', '--', '-.', ':']
MARKERS     = ['o', 's', '^', 'D', 'v', '<', '>', '1', '2', '3']
COLORS      = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
               'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

LABEL_FS  = 14   # axis label
TICK_FS   = 12   # tick / annotation
LEGEND_FS = 11

HERE        = os.path.dirname(os.path.abspath(__file__))
PAPERS      = os.path.join(HERE, "..", "..", "papers")
TAPERKV_FIG = os.path.join(PAPERS, "taperkv_paper", "figures")
HALO_FIG    = os.path.join(PAPERS, "halo_paper",    "figures")
os.makedirs(TAPERKV_FIG, exist_ok=True)
os.makedirs(HALO_FIG,    exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────
def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)["results"]


def save_fig(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {path}")


def _style_ax(ax: plt.Axes, xlabel: str, ylabel: str, ylim=None) -> None:
    ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.5, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _model_tag(ax: plt.Axes, text: str) -> None:
    """Place model name in top-left corner of the axes."""
    ax.text(0.03, 0.97, text, transform=ax.transAxes,
            fontsize=TICK_FS, va='top', ha='left',
            fontstyle='italic')


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Figure 1 – b_min sweep (TaperKV key-floor sensitivity)                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def fig_bmin_sweep() -> None:
    datasets = [
        (load(os.path.join(HERE, "results_ablation", "bmin_llama",
                           "results_llama-3.1-8b-instruct.json")),
         "Llama-3.1-8B"),
        (load(os.path.join(HERE, "results_ablation", "bmin_r1q7",
                           "results_DeepSeek-R1-Distill-Qwen-7B.json")),
         "R1-Distill-Qwen-7B"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)

    bmin3 = [2, 3, 4, 5]
    bmin4 = [2, 4, 6]

    for ax, (results, model_name) in zip(axes, datasets):
        fkv  = results["FullKV"]["needle"]["accuracy"]
        # Reference codec line: HayateQuant-3bit (proposed); fall back to the
        # legacy HALO-3bit label for old result JSONs.
        ref_label = "HayateQuant-3bit" if "HayateQuant-3bit" in results else "HALO-3bit"
        ref = results.get(ref_label, {}).get("needle", {}).get("accuracy")

        # reference horizontal lines (plotted behind data)
        ax.axhline(fkv, color='black', linestyle=':', linewidth=1.4, zorder=1,
                   label=f"FullKV  ({fkv:.3f})")
        if ref is not None:
            ax.axhline(ref, color=COLORS[3], linestyle='--', linewidth=1.2, zorder=1,
                       label=f"{ref_label}  ({ref:.3f})")

        # b_min sweep lines
        vals3 = [results.get(f"TaperKV-3bit-bmin{b}", {}).get("needle", {})
                             .get("accuracy", np.nan) for b in bmin3]
        vals4 = [results.get(f"TaperKV-4bit-bmin{b}", {}).get("needle", {})
                             .get("accuracy", np.nan) for b in bmin4]

        ax.plot(bmin3, vals3,
                color=COLORS[0], linestyle=LINE_STYLES[0], marker=MARKERS[0],
                linewidth=1.8, markersize=7, zorder=3,
                label=r"TaperKV-3bit ($b_{\min}$ sweep)")
        ax.plot(bmin4, vals4,
                color=COLORS[1], linestyle=LINE_STYLES[1], marker=MARKERS[1],
                linewidth=1.8, markersize=7, zorder=3,
                label=r"TaperKV-4bit ($b_{\min}$ sweep)")

        # collapse boundary marker
        ax.axvline(2.5, color='gray', linestyle=':', linewidth=0.8, zorder=0)

        _style_ax(ax,
                  xlabel=r"Key bit-width floor $b_{\min}$",
                  ylabel="Needle Accuracy",
                  ylim=(-0.05, 1.18))
        ax.set_xticks([2, 3, 4, 5, 6])
        ax.set_xlim(1.6, 6.5)
        _model_tag(ax, model_name)
        ax.legend(fontsize=LEGEND_FS, loc='lower right',
                  borderpad=0.5, labelspacing=0.3)

    save_fig(fig, os.path.join(TAPERKV_FIG, "fig_bmin_sweep.png"))


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Figure 2 – Memory-budget trade-off                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def fig_budget_sweep() -> None:
    budgets = [10, 15, 20, 25, 35]

    method_order = [
        ("StreamingLLM",  "StreamingLLM"),
        ("ThinKV",        "ThinKV"),
        ("KIVI-INT2",     "KIVI-INT2"),
        ("HayateQuant-3bit", "HayateQuant-3bit"),
        ("HALO-3bit",     "HALO-3bit"),
        ("TaperKV-3bit",  "TaperKV-3bit"),
        ("TaperKV",       "TaperKV"),
    ]

    def load_budget_series(prefix: str):
        out: dict[str, dict[int, float]] = {}
        for p in budgets:
            pat = os.path.join(HERE, "results_ablation", f"budget_{prefix}_{p}",
                               "results_*.json")
            files = glob.glob(pat)
            if not files:
                continue
            for m, v in load(files[0]).items():
                out.setdefault(m, {})[p] = v.get("needle", {}).get("accuracy", np.nan)
        return out

    llama_data = load_budget_series("llama")
    r1q7_data  = load_budget_series("r1q7")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.subplots_adjust(bottom=0.28, wspace=0.32)

    handles_all, labels_all = [], []

    for ax, (data, model_name) in zip(axes, [
            (llama_data, "Llama-3.1-8B"),
            (r1q7_data,  "R1-Distill-Qwen-7B")]):

        # FullKV reference
        fkv_vals = [data.get("FullKV", {}).get(p, np.nan) for p in budgets]
        h, = ax.plot(budgets, fkv_vals, color='black', linestyle=':',
                     linewidth=1.4, label="FullKV", zorder=1)
        if ax is axes[0]:
            handles_all.append(h); labels_all.append("FullKV")

        for idx, (key, label) in enumerate(method_order):
            if key not in data:
                continue
            ys = [data[key].get(p, np.nan) for p in budgets]
            h, = ax.plot(budgets, ys,
                         color=COLORS[idx % len(COLORS)],
                         linestyle=LINE_STYLES[idx % len(LINE_STYLES)],
                         marker=MARKERS[idx % len(MARKERS)],
                         linewidth=1.8, markersize=6, zorder=3, label=label)
            if ax is axes[0]:
                handles_all.append(h); labels_all.append(label)

        _style_ax(ax,
                  xlabel="Target Memory Budget (%)",
                  ylabel="Needle Accuracy",
                  ylim=(-0.05, 1.15))
        ax.set_xticks(budgets)
        ax.set_xticklabels([f"{b}%" for b in budgets], fontsize=TICK_FS)
        ax.set_xlim(8, 37)
        _model_tag(ax, model_name)

    # shared legend below both subplots — avoids overlapping with flat lines at 1.0
    fig.legend(handles_all, labels_all,
               loc='lower center', ncol=4,
               fontsize=LEGEND_FS, framealpha=1,
               edgecolor='black', fancybox=False,
               bbox_to_anchor=(0.5, 0.01),
               borderpad=0.5, labelspacing=0.3, columnspacing=1.0)

    save_fig(fig, os.path.join(TAPERKV_FIG, "fig_budget_sweep.png"))


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Figure 3 – TaperKV K/V bit-asymmetry (Needle + PPL, Ministral + Llama)    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def fig_kv_asymmetry() -> None:
    datasets = [
        (load(os.path.join(HERE, "results_ablation", "halo_minis",
                           "results_Ministral-8B-Instruct-2410.json")),
         "Ministral-8B"),
        (load(os.path.join(HERE, "results_ablation", "halo_llama",
                           "results_llama-3.1-8b-instruct.json")),
         "Llama-3.1-8B"),
    ]

    kv_methods = [
        ("TaperKV-asym",   r"$K$$>$$V$",  COLORS[0]),
        ("TaperKV-sym",    r"$K$$=$$V$",  COLORS[1]),
        ("TaperKV-revsym", r"$V$$>$$K$",  COLORS[2]),
    ]
    x = np.arange(len(kv_methods))
    bar_w = 0.55

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), constrained_layout=True)

    for col, (results, model_name) in enumerate(datasets):
        fkv_needle = results["FullKV"]["needle"]["accuracy"]
        fkv_ppl    = results["FullKV"]["ppl"]["ppl"]

        needles = [results.get(m, {}).get("needle", {}).get("accuracy", 0)
                   for m, _, _ in kv_methods]
        ppls    = [results.get(m, {}).get("ppl", {}).get("ppl", 0)
                   for m, _, _ in kv_methods]

        # ── top: Needle ────────────────────────────────────────────────────
        ax_n = axes[0, col]
        for i, ((_, xlbl, c), val) in enumerate(zip(kv_methods, needles)):
            ax_n.bar(x[i], val, width=bar_w, color=c, alpha=0.85,
                     edgecolor='black', linewidth=0.7, zorder=3)
        ax_n.axhline(fkv_needle, color='black', linestyle='--',
                     linewidth=1.4, zorder=4, label=f"FullKV ({fkv_needle:.2f})")

        # value labels — only if bar height ≥ 0.06
        for i, val in enumerate(needles):
            if val >= 0.06:
                ax_n.text(x[i], val + 0.025, f"{val:.2f}",
                          ha='center', va='bottom', fontsize=10, fontweight='bold')

        _style_ax(ax_n, xlabel="", ylabel="Needle Accuracy", ylim=(0, 1.22))
        ax_n.set_xticks(x)
        ax_n.set_xticklabels([xlbl for _, xlbl, _ in kv_methods], fontsize=TICK_FS)
        ax_n.legend(fontsize=LEGEND_FS, loc='upper right',
                    borderpad=0.4, labelspacing=0.2)
        _model_tag(ax_n, model_name)

        # ── bottom: PPL ────────────────────────────────────────────────────
        ax_p = axes[1, col]
        for i, ((_, xlbl, c), val) in enumerate(zip(kv_methods, ppls)):
            ax_p.bar(x[i], val, width=bar_w, color=c, alpha=0.85,
                     edgecolor='black', linewidth=0.7, zorder=3)
        ax_p.axhline(fkv_ppl, color='black', linestyle='--',
                     linewidth=1.4, zorder=4, label=f"FullKV ({fkv_ppl:.2f})")

        ppl_max = max(ppls) * 1.15 if ppls else fkv_ppl * 1.5
        for i, val in enumerate(ppls):
            if val > 0:
                ax_p.text(x[i], val + ppl_max * 0.012, f"{val:.1f}",
                          ha='center', va='bottom', fontsize=10, fontweight='bold')

        _style_ax(ax_p, xlabel="", ylabel="Perplexity ↓",
                  ylim=(0, ppl_max))
        ax_p.set_xticks(x)
        ax_p.set_xticklabels([xlbl for _, xlbl, _ in kv_methods], fontsize=TICK_FS)
        ax_p.legend(fontsize=LEGEND_FS, loc='upper left',
                    borderpad=0.4, labelspacing=0.2)

    save_fig(fig, os.path.join(TAPERKV_FIG, "fig_kv_asymmetry.png"))


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Figure 4 – K/V asymmetry on reasoning tasks (GSM8K + MATH500)             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def fig_kv_asymmetry_reason() -> None:
    datasets = [
        (os.path.join(HERE, "results_ablation", "reason_r1q7",
                      "results_DeepSeek-R1-Distill-Qwen-7B.json"),
         "R1-Distill-Qwen-7B"),
        (os.path.join(HERE, "results_ablation", "reason_r1l8",
                      "results_DeepSeek-R1-Distill-Llama-8B.json"),
         "R1-Distill-Llama-8B"),
    ]

    methods = [
        ("TaperKV-asym",   r"$K$$>$$V$",  COLORS[0]),
        ("TaperKV-sym",    r"$K$$=$$V$",  COLORS[1]),
        ("TaperKV-revsym", r"$V$$>$$K$",  COLORS[2]),
        ("TurboQuant-3bit", "TurboQ-3bit", COLORS[3]),
    ]
    x    = np.arange(len(methods))
    bar_w = 0.55

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), constrained_layout=True)

    for col, (path, model_name) in enumerate(datasets):
        r = load(path)
        fkv_gsm  = r["FullKV"]["gsm8k"]["pass@1"]
        fkv_math = r["FullKV"]["math500"]["pass@1"]

        gsm_vals  = [r.get(m, {}).get("gsm8k",  {}).get("pass@1", 0) for m, _, _ in methods]
        math_vals = [r.get(m, {}).get("math500", {}).get("pass@1", 0) for m, _, _ in methods]

        for row, (vals, fkv_val, ylabel, task) in enumerate([
                (gsm_vals,  fkv_gsm,  "GSM8K Pass@1",  "GSM8K"),
                (math_vals, fkv_math, "MATH500 Pass@1", "MATH500"),
        ]):
            ax = axes[row, col]
            for i, ((_, xlbl, c), val) in enumerate(zip(methods, vals)):
                ax.bar(x[i], val, width=bar_w, color=c, alpha=0.85,
                       edgecolor='black', linewidth=0.7, zorder=3)
            ax.axhline(fkv_val, color='black', linestyle='--',
                       linewidth=1.4, zorder=4, label=f"FullKV ({fkv_val:.2f})")

            top = max(fkv_val, max(vals) if vals else 0) * 1.22
            for i, val in enumerate(vals):
                if val >= 0.04:
                    ax.text(x[i], val + top * 0.012, f"{val:.2f}",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

            _style_ax(ax, xlabel="", ylabel=ylabel, ylim=(0, top))
            ax.set_xticks(x)
            ax.set_xticklabels([xlbl for _, xlbl, _ in methods],
                               fontsize=TICK_FS)
            ax.legend(fontsize=LEGEND_FS, loc='upper right',
                      borderpad=0.4, labelspacing=0.2)
            if row == 0:
                _model_tag(ax, model_name)

    save_fig(fig, os.path.join(TAPERKV_FIG, "fig_kv_asymmetry_reason.png"))


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Figure 5 – HALO lever PPL (Ministral + Llama, 3-bit + 4-bit)              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def fig_halo_lever() -> None:
    datasets = [
        (load(os.path.join(HERE, "results_ablation", "halo_minis",
                           "results_Ministral-8B-Instruct-2410.json")),
         "Ministral-8B"),
        (load(os.path.join(HERE, "results_ablation", "halo_llama",
                           "results_llama-3.1-8b-instruct.json")),
         "Llama-3.1-8B"),
    ]

    lever_cfg = {
        3: [
            ("TurboQuant-3bit", "TurboQ",      COLORS[0]),
            ("HALO-h-noLS-3",   "h-noLS",      COLORS[1]),
            ("HALO-hg-noLS-3",  "hg-noLS",     COLORS[2]),
            ("HALO-hg-LS-3",    "hg-LS\n(HALO)",COLORS[3]),
            ("HALO-dense-LS-3", "dense-LS",    COLORS[4]),
        ],
        4: [
            ("TurboQuant-4bit", "TurboQ",      COLORS[0]),
            ("HALO-h-noLS-4",   "h-noLS",      COLORS[1]),
            ("HALO-hg-noLS-4",  "hg-noLS",     COLORS[2]),
            ("HALO-hg-LS-4",    "hg-LS\n(HALO)",COLORS[3]),
            ("HALO-dense-LS-4", "dense-LS",    COLORS[4]),
        ],
    }
    x = np.arange(5)
    bar_w = 0.6

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), constrained_layout=True)

    for row, bits in enumerate([3, 4]):
        for col, (results, model_name) in enumerate(datasets):
            ax = axes[row, col]
            cfg = lever_cfg[bits]
            fkv = results["FullKV"]["ppl"]["ppl"]
            ppls = [results.get(m, {}).get("ppl", {}).get("ppl", 0)
                    for m, _, _ in cfg]

            for i, ((_, xlbl, c), val) in enumerate(zip(cfg, ppls)):
                ax.bar(x[i], val, width=bar_w, color=c, alpha=0.85,
                       edgecolor='black', linewidth=0.7, zorder=3)

            ax.axhline(fkv, color='black', linestyle='--', linewidth=1.4,
                       zorder=4, label=f"FullKV ({fkv:.2f})")

            # tight y-axis to show small differences clearly
            ppl_range = max(ppls) - min(ppls) if ppls else 0.1
            y_lo = max(0, fkv - ppl_range * 0.5)
            y_hi = max(ppls) * 1.0 + ppl_range * 1.5 if ppls else fkv * 1.2

            _style_ax(ax, xlabel="", ylabel=f"Perplexity ↓  ({bits}-bit)",
                      ylim=(y_lo, y_hi))
            ax.set_xticks(x)
            ax.set_xticklabels([xlbl for _, xlbl, _ in cfg], fontsize=TICK_FS - 1)
            ax.legend(fontsize=LEGEND_FS, loc='upper left',
                      borderpad=0.4, labelspacing=0.2)
            if row == 0:
                _model_tag(ax, model_name)

    save_fig(fig, os.path.join(HALO_FIG, "fig_halo_lever.png"))


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Figure 6 – HALO lever on reasoning models (3-bit PPL, R1-L8 + R1-Q7)      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def fig_halo_lever_reasoning() -> None:
    datasets = [
        (load(os.path.join(HERE, "results_ablation", "halo_r1l8",
                           "results_DeepSeek-R1-Distill-Llama-8B.json")),
         "R1-Distill-Llama-8B"),
        (load(os.path.join(HERE, "results_ablation", "halo_r1q7",
                           "results_DeepSeek-R1-Distill-Qwen-7B.json")),
         "R1-Distill-Qwen-7B"),
    ]

    lever_methods = [
        ("TurboQuant-3bit", "TurboQ",      COLORS[0]),
        ("HALO-h-noLS-3",   "h-noLS",      COLORS[1]),
        ("HALO-hg-noLS-3",  "hg-noLS",     COLORS[2]),
        ("HALO-hg-LS-3",    "hg-LS\n(HALO)",COLORS[3]),
        ("HALO-dense-LS-3", "dense-LS",    COLORS[4]),
    ]
    x = np.arange(len(lever_methods))
    bar_w = 0.6

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)

    for ax, (results, model_name) in zip(axes, datasets):
        fkv = results["FullKV"]["ppl"]["ppl"]
        ppls = [results.get(m, {}).get("ppl", {}).get("ppl", 0)
                for m, _, _ in lever_methods]

        for i, ((_, xlbl, c), val) in enumerate(zip(lever_methods, ppls)):
            ax.bar(x[i], val, width=bar_w, color=c, alpha=0.85,
                   edgecolor='black', linewidth=0.7, zorder=3)

        ax.axhline(fkv, color='black', linestyle='--', linewidth=1.4,
                   zorder=4, label=f"FullKV ({fkv:.2f})")

        ppl_range = max(ppls) - min(ppls) if ppls else 0.5
        y_lo = max(0, fkv - ppl_range * 0.3)
        y_hi = max(ppls) * 1.0 + ppl_range * 2.0 if ppls else fkv * 1.3

        _style_ax(ax, xlabel="", ylabel="Perplexity ↓  (3-bit)",
                  ylim=(y_lo, y_hi))
        ax.set_xticks(x)
        ax.set_xticklabels([xlbl for _, xlbl, _ in lever_methods], fontsize=TICK_FS)
        ax.legend(fontsize=LEGEND_FS, loc='upper left',
                  borderpad=0.4, labelspacing=0.2)
        _model_tag(ax, model_name)

    save_fig(fig, os.path.join(HALO_FIG, "fig_halo_lever_reasoning.png"))


if __name__ == "__main__":
    print("Generating ablation figures …")
    fig_bmin_sweep()
    fig_budget_sweep()
    fig_kv_asymmetry()
    fig_kv_asymmetry_reason()
    fig_halo_lever()
    fig_halo_lever_reasoning()
    print("Done.")
