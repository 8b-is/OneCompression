"""Generate the quantitative figures for the HayateQuant paper.

All numbers are copied verbatim from measured results recorded in
papers/hayatequant_paper/README.md (onecomp-lab, branch lab/kvcomp):
  - 3-way long-context protocol (job 59004, Llama-3.1-8B-Instruct)
  - e2e speed v10 final campaign (prompt=4096, gen=128)
  - lever-isolation ablation (job 59050)
  - v8 K/V-asymmetric bits + calibration (jobs 59102-59125)

Style: house academic template (sans-serif text, STIX mathtext, inward
ticks, boxed legends); colors/markers/linestyles are drawn strictly from the
fixed palettes below, in order, so every figure in the paper is visually
consistent. In-plot annotation text is avoided wherever the same information
can be conveyed by the legend or the caption instead.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.xmargin'] = 0.01
plt.rcParams['axes.ymargin'] = 0.01
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.edgecolor"] = 'none'
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

line_styles_list = ['-', '--', '-.', ':']
makers_list = ['o', 's', '^', 'D', 'v', '<', '>', '1', '2', '3']
color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray']

# Fixed roles used consistently across every figure in the paper: the
# baseline is blue, HayateQuant (proposed) is red, and a neutral gray marks
# reference lines or rejected/ablated-away variants.
C_TQ = color_list[0]
C_HQ = color_list[3]
C_REF = color_list[7]


def despine(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def scatter_series(ax, points, *, color, label, ms=6, mfc=None, mec=None, mew=1.2,
                   zorder=4, bit_markers=None):
    """Plot a scatter series; optional per-point markers encode bit width."""
    xs, ys = zip(*points)
    if bit_markers is None:
        ax.plot(xs, ys, marker=makers_list[0], linestyle="none", ms=ms,
                color=color, markerfacecolor=mfc or color,
                markeredgecolor=mec or color, markeredgewidth=mew,
                label=label, zorder=zorder)
        return
    for (x, y), mk in zip(points, bit_markers):
        ax.plot(x, y, marker=mk, linestyle="none", ms=ms,
                color=color, markerfacecolor=mfc or color,
                markeredgecolor=mec or color, markeredgewidth=mew,
                zorder=zorder)
    ax.plot([], [], marker=bit_markers[1], linestyle="none", ms=ms,
            color=color, markerfacecolor=mfc or color,
            markeredgecolor=mec or color, markeredgewidth=mew,
            label=label, zorder=zorder)


def style_legend(ax, **kwargs):
    leg = ax.legend(**kwargs)
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_edgecolor("none")
    return leg

# ---------------------------------------------------------------- Figure 1
# Left: PPL vs avg bits (same protocol: kv_benchmark, WikiText-2, 16 windows)
bits = [2.0, 2.5, 3.0, 3.5, 4.0]
ppl_tq = [10.168, 7.772, 6.917, 6.696, 6.565]
ppl_hq = [9.046, 7.192, 6.810, 6.631, 6.569]
ppl_full = 6.486

# Right: speed-quality trade-off (e2e decode tok/s from the v10 single-node
# same-session A/B campaign, run_rt_speed_v10_final.sh: TQ under its published
# butterfly implementation (session A) and under the Dao-AILab FWHT CUDA
# kernel (session B), HQ under the fused Triton+CUDA-graph serving path
# (session B); PPL from the long-context protocol). Ordered 2, 3, 4 bits.
tq_sq = [(20.3, 10.168), (20.3, 6.917), (20.4, 6.565)]
tqk_sq = [(33.9, 10.168), (34.3, 6.917), (33.6, 6.565)]
hq_sq = [(48.8, 9.046), (48.7, 6.810), (48.7, 6.569)]
full_tps = 75.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 2.95),
                               gridspec_kw={"width_ratios": [1.15, 1.0]})
ax1.plot(bits, ppl_tq, marker=makers_list[0], linestyle=line_styles_list[0],
          color=C_TQ, label="TurboQuant", lw=1.8, ms=5)
ax1.plot(bits, ppl_hq, marker=makers_list[3], linestyle=line_styles_list[2],
          color=C_HQ, label="HayateQuant", lw=2.0, ms=5.5)
ax1.axhline(ppl_full, color=C_REF, ls=line_styles_list[3], lw=2.0,
            label=f"FullKV ({ppl_full:.3f})", zorder=2)
ax1.set_xlabel("average bits / coordinate")
ax1.set_ylabel("WikiText-2 perplexity")
ax1.set_title("Quality at equal stored bits")
style_legend(ax1, loc="upper right")
ax1.grid(alpha=0.25, lw=0.5)
despine(ax1)

# faint iso-bit connectors across the three variants, same bit-width, drawn
# left to right (TurboQuant butterfly -> +FWHT kernel -> HayateQuant) so the
# lines never cross back over a marker; top-to-bottom the three dashed lines
# are 2, 3, and 4 bits (explained in the caption rather than annotated in
# the plot).
# Marker shape encodes bit width (2 / 3 / 4) on the speed panel so FWHT and
# HayateQuant points at nearly the same throughput do not occlude each other.
BIT_MK = [makers_list[0], makers_list[1], makers_list[3]]  # circle, square, diamond

for (xt, yt), (xk, yk), (xh, yh) in zip(tq_sq, tqk_sq, hq_sq):
    ax2.plot([xt, xk, xh], [yt, yk, yh], ls=line_styles_list[1], lw=0.8,
             color="#c3cad2", zorder=1)
scatter_series(ax2, tq_sq, color=C_TQ, label="TurboQuant (butterfly)",
               ms=6, bit_markers=BIT_MK, zorder=4)
scatter_series(ax2, tqk_sq, color=C_TQ, label="TurboQuant (+FWHT kernel)",
               ms=5.8, mfc="white", mec=C_TQ, mew=1.5,
               bit_markers=BIT_MK, zorder=6)
scatter_series(ax2, hq_sq, color=C_HQ, label="HayateQuant",
               ms=6, bit_markers=BIT_MK, zorder=5)
ax2.axhline(ppl_full, color=C_REF, ls=line_styles_list[3], lw=2.0,
            label=f"FullKV ({full_tps:.1f} tok/s)", zorder=2)
ax2.set_xlim(14.5, 56.5)
ax2.set_ylim(6.05, 10.65)
ax2.set_xlabel("end-to-end decode throughput (tok/s)")
ax2.set_ylabel("WikiText-2 perplexity")
ax2.set_title("Speed-quality trade-off")
style_legend(ax2, loc="upper right", fontsize=7.6)
ax2.grid(alpha=0.25, lw=0.5)
despine(ax2)
fig.tight_layout()
fig.savefig("figures/teaser.pdf")
fig.savefig("figures/teaser_preview.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Figure: ablation
labels = [
    "TurboQuant (baseline)",
    "$b{=}1$ (scalar, role-asym. scales)",
    "$b{=}2$ block-VQ",
    "full ($b{=}4$ block-VQ)",
    "$-$ role-asym. scales",
    "$-$ hybrid rotation",
]
vals = [10.161, 10.022, 9.512, 9.147, 9.198, 9.178]
cols = [color_list[0], color_list[1], color_list[2], color_list[3],
        color_list[4], color_list[4]]

fig, ax = plt.subplots(figsize=(6.0, 2.6))
y = range(len(labels))[::-1]
ax.barh(list(y), vals, color=cols, height=0.62)
ax.set_yticks(list(y))
ax.set_yticklabels(labels)
ax.set_xlim(8.9, 10.45)
ax.set_xlabel("WikiText-2 perplexity (2 bits, Llama-3.1-8B)")
ax.grid(alpha=0.25, lw=0.5, axis="x")
despine(ax)
fig.savefig("figures/ablation.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure: asym + calib
fig, ax = plt.subplots(figsize=(4.8, 2.55))
groups = ["Llama-3.1-8B", "Qwen3-8B"]
k3v3 = [7.020, 8.800]
k4v2 = [6.933, 8.648]
combo = [6.922, 8.660]
k2v4 = [8.444, 9.715]
x = [0, 1]
w = 0.19
for i, (vals_, lab, c) in enumerate([
        (k3v3, "uniform K3V3", color_list[0]),
        (k4v2, "K4V2", color_list[1]),
        (combo, "K4V2 + calibration", color_list[3]),
        (k2v4, "K2V4 (reversed)", color_list[7])]):
    xs = [xi + (i - 1.5) * w for xi in x]
    ax.bar(xs, vals_, width=w * 0.92, color=c, label=lab)
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylabel("perplexity @ 3.0 avg bits")
ax.set_ylim(6.4, 10.3)
style_legend(ax, ncol=2, fontsize=7.6, loc="upper left")
ax.grid(alpha=0.25, lw=0.5, axis="y")
despine(ax)
fig.savefig("figures/asym_calib.pdf")
plt.close(fig)

print("figures written")
