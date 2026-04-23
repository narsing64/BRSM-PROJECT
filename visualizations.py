"""
=============================================================
  Selective Attention Study — Report 1 Visualizations
  (CORRECTED VERSION)
=============================================================
BUGS FIXED vs original:
  BUG V1 — Fig 4: annotation always showed Pearson r even for
            the single group where Spearman is the primary test
            (lab RT is non-normal). FIX: check normality and
            display the correct statistic per group.

  BUG V2 — Fig 7: t-test ran on raw trial data (555 rows).
            Trials from the same participant are NOT independent
            — this is pseudoreplication that artificially inflates
            degrees of freedom and shrinks p-values.
            FIX: aggregate to participant-level means first
            (one red mean and one white mean per participant),
            then run the test on those 21 / 16 means.

  BUG V3 — Fig 7: used ttest_ind (independent samples).
            The same participant contributes BOTH a red-target
            mean and a white-target mean → within-person
            comparison. FIX: use ttest_rel (paired t-test).
            For the multiple group where difference scores are
            non-normal, also report Wilcoxon signed-rank.

PLOTS GENERATED (saved to processed_data/figures/):
  Fig 1 — Histograms: RT distributions per condition
  Fig 2 — Boxplots: RT across all 4 cells (2×2)
  Fig 3 — Bar chart: Mean RT ± 95% CI per condition
  Fig 4 — Scatter: Lab RT vs Game RT per group (RQ1)
  Fig 5 — Bar chart: Accuracy across 4 cells
  Fig 6 — Line plot: Mean RT per game level (RQ4)
  Fig 7 — Boxplot: RT by target colour × group (lab)
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import os
import warnings
warnings.filterwarnings("ignore")

INPUT_DIR = "processed_data"
FIG_DIR   = "processed_data/figures"
os.makedirs(FIG_DIR, exist_ok=True)

long   = pd.read_csv(f"{INPUT_DIR}/master_dataset_long.csv")
wide   = pd.read_csv(f"{INPUT_DIR}/master_dataset_wide.csv")
levels = pd.read_csv(f"{INPUT_DIR}/game_levels_long.csv")
trials = pd.read_csv(f"{INPUT_DIR}/lab_trials_long.csv")

C = {
    "single_lab"  : "#2E86AB",
    "single_game" : "#A8DADC",
    "multi_lab"   : "#E63946",
    "multi_game"  : "#F4A261",
    "single"      : "#2E86AB",
    "multiple"    : "#E63946",
    "lab"         : "#457B9D",
    "game"        : "#E76F51",
}

CELL_COLORS = {
    ("single",   "lab") : C["single_lab"],
    ("single",   "game"): C["single_game"],
    ("multiple", "lab") : C["multi_lab"],
    ("multiple", "game"): C["multi_game"],
}

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "font.size"        : 11,
    "axes.titlesize"   : 13,
    "axes.titleweight" : "bold",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "figure.dpi"       : 150,
})


# ─────────────────────────────────────────────────────────────
# FIG 1 — Histograms: RT distributions per condition
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Figure 1: Reaction Time Distributions per Condition",
             fontsize=15, fontweight="bold", y=1.01)

cells  = [("single","lab"), ("single","game"), ("multiple","lab"), ("multiple","game")]
titles = ["Single Target — Lab", "Single Target — Game",
          "Multiple Target — Lab", "Multiple Target — Game"]

for ax, (grp, mod), title in zip(axes.flat, cells, titles):
    sub = long[(long["group"]==grp) & (long["modality"]==mod)]["RT_ms"]
    col = CELL_COLORS[(grp, mod)]

    ax.hist(sub, bins=8, color=col, edgecolor="white", alpha=0.85, linewidth=1.2)

    x = np.linspace(sub.min(), sub.max(), 200)
    mu, sigma = sub.mean(), sub.std()
    y = stats.norm.pdf(x, mu, sigma) * len(sub) * (sub.max()-sub.min()) / 8
    ax.plot(x, y, color="black", linewidth=1.8, linestyle="--", alpha=0.7)

    w, p = stats.shapiro(sub)
    norm_label = "Normal" if p > 0.05 else "Non-normal"
    ax.set_title(f"{title}\nM={mu:.0f}, SD={sigma:.0f}", fontsize=11, fontweight="bold")
    ax.set_xlabel("RT (ms)")
    ax.set_ylabel("Frequency")
    ax.annotate(f"Shapiro-Wilk: W={w:.3f}, p={p:.3f}\n({norm_label})",
                xy=(0.97, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.8))

fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig1_rt_histograms.png", bbox_inches="tight")
plt.close()
print("✅ Fig 1 saved — RT histograms")


# ─────────────────────────────────────────────────────────────
# FIG 2 — Boxplots: RT across all 4 cells
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("Figure 2: Reaction Time Distribution Across Conditions",
             fontsize=15, fontweight="bold")

for ax, grp in zip(axes, ["single", "multiple"]):
    lab_data  = long[(long["group"]==grp) & (long["modality"]=="lab")]["RT_ms"]
    game_data = long[(long["group"]==grp) & (long["modality"]=="game")]["RT_ms"]

    bp = ax.boxplot(
        [lab_data, game_data],
        labels=["Lab", "Game"],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markerfacecolor="gray", markersize=7, alpha=0.6),
        widths=0.5,
    )
    for patch, col in zip(bp["boxes"], [CELL_COLORS[(grp,"lab")], CELL_COLORS[(grp,"game")]]):
        patch.set_facecolor(col)
        patch.set_alpha(0.8)

    for i, data in enumerate([lab_data, game_data], 1):
        jitter = np.random.normal(0, 0.05, len(data))
        ax.scatter(np.full(len(data), i) + jitter, data,
                   color="black", alpha=0.4, s=30, zorder=3)

    n = long[long["group"]==grp]["participant"].nunique()
    ax.set_title(f"{'Single' if grp=='single' else 'Multiple'} Target Group (n={n})",
                 fontweight="bold")
    ax.set_ylabel("Reaction Time (ms)")
    ax.set_xlabel("Modality")

fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig2_rt_boxplots.png", bbox_inches="tight")
plt.close()
print("✅ Fig 2 saved — RT boxplots")


# ─────────────────────────────────────────────────────────────
# FIG 3 — Bar chart: Mean RT ± 95% CI (2×2 design)
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Figure 3: Mean Reaction Time by Group and Modality (±95% CI)",
             fontsize=15, fontweight="bold")

groups     = ["single", "multiple"]
modalities = ["lab", "game"]
x          = np.array([0, 1])
bar_width  = 0.35
offsets    = [-bar_width/2, bar_width/2]

for i, mod in enumerate(modalities):
    means, cis, ns = [], [], []
    for grp in groups:
        sub = long[(long["group"]==grp) & (long["modality"]==mod)]["RT_ms"]
        n   = len(sub)
        m   = sub.mean()
        se  = sub.std(ddof=1) / np.sqrt(n)
        ci  = stats.t.ppf(0.975, df=n-1) * se
        means.append(m); cis.append(ci); ns.append(n)

    bars = ax.bar(x + offsets[i], means, width=bar_width,
                  color=[C["lab"] if mod=="lab" else C["game"]] * 2,
                  alpha=0.85, edgecolor="white", linewidth=1.2,
                  label=f"{'Lab Task' if mod=='lab' else 'Game'}")
    ax.errorbar(x + offsets[i], means, yerr=cis,
                fmt="none", color="black", capsize=6, linewidth=2, capthick=2)

    for j, (bar, m, n) in enumerate(zip(bars, means, ns)):
        ax.text(bar.get_x() + bar.get_width()/2, m + 50,
                f"{m:.0f}\n(n={n})", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(["Single Target\n(n=21)", "Multiple Target\n(n=16)"], fontsize=12)
ax.set_ylabel("Mean Reaction Time (ms)", fontsize=12)
ax.set_xlabel("Target Load Group", fontsize=12)
ax.legend(fontsize=11, framealpha=0.9)
ax.set_ylim(0, max(long["RT_ms"].max() * 1.25, 4500))

fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig3_rt_barchart.png", bbox_inches="tight")
plt.close()
print("✅ Fig 3 saved — Mean RT bar chart")


# ─────────────────────────────────────────────────────────────
# FIG 4 — Scatter: Lab RT vs Game RT (RQ1 — Concurrent Validity)
# BUG V1 FIX: check normality per group and display correct statistic
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("Figure 4: Concurrent Validity — Lab RT vs Game RT (RQ1)",
             fontsize=15, fontweight="bold")

for ax, grp in zip(axes, ["single", "multiple"]):
    sub = wide[wide["group"] == grp]
    x   = sub["lab_RT"].values
    y   = sub["game_RT"].values
    col = C[grp]

    # ── BUG V1 FIX: choose primary statistic based on normality ──
    _, p_norm_x = stats.shapiro(x)
    _, p_norm_y = stats.shapiro(y)
    both_normal = (p_norm_x > 0.05) and (p_norm_y > 0.05)

    r_p, p_p = stats.pearsonr(x, y)
    r_s, p_s = stats.spearmanr(x, y)

    if both_normal:
        r_use, p_use, stat_name = r_p, p_p, "Pearson r"
    else:
        r_use, p_use, stat_name = r_s, p_s, "Spearman rho"

    ax.scatter(x, y, color=col, s=70, alpha=0.75,
               edgecolors="white", linewidths=0.8, zorder=3)

    m_fit, b_fit = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min()-100, x.max()+100, 100)
    ax.plot(x_line, m_fit*x_line + b_fit,
            color=col, linewidth=2, linestyle="--", alpha=0.8, label="Regression fit")

    lim = [min(x.min(), y.min()) - 200, max(x.max(), y.max()) + 200]
    ax.plot(lim, lim, color="gray", linewidth=1.2,
            linestyle=":", alpha=0.6, label="Perfect agreement (y=x)")

    sig_str = "p < .05" if p_use < 0.05 else f"p = {p_use:.3f}"
    ax.annotate(
        f"{stat_name} = {r_use:.3f}\n{sig_str}\n(n = {len(sub)})",
        xy=(0.05, 0.92), xycoords="axes fraction",
        fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="gray", alpha=0.9)
    )

    grp_label = "Single Target" if grp == "single" else "Multiple Target"
    ax.set_title(f"{grp_label} Group", fontweight="bold")
    ax.set_xlabel("Lab RT (ms)", fontsize=11)
    ax.set_ylabel("Game RT (ms)", fontsize=11)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.legend(fontsize=9, loc="lower right")

fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig4_validity_scatter.png", bbox_inches="tight")
plt.close()
print("✅ Fig 4 saved — Validity scatter plot")


# ─────────────────────────────────────────────────────────────
# FIG 5 — Bar chart: Accuracy across 4 cells
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("Figure 5: Mean Accuracy by Group and Modality (±95% CI)",
             fontsize=15, fontweight="bold")

for ax, grp in zip(axes, ["single", "multiple"]):
    sub_lab  = long[(long["group"]==grp)&(long["modality"]=="lab")]["accuracy"]
    sub_game = long[(long["group"]==grp)&(long["modality"]=="game")]["accuracy"]

    means, cis, labels, colors = [], [], [], []
    for mod, sub in [("Lab", sub_lab), ("Game", sub_game)]:
        n  = len(sub)
        m  = sub.mean()
        se = sub.std(ddof=1) / np.sqrt(n) if sub.std() > 0 else 0
        ci = stats.t.ppf(0.975, df=n-1) * se if n > 1 else 0
        means.append(m); cis.append(ci)
        labels.append(mod)
        colors.append(C["lab"] if mod=="Lab" else C["game"])

    bars = ax.bar(labels, means, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=1.2, width=0.5)
    ax.errorbar(labels, means, yerr=cis,
                fmt="none", color="black", capsize=6, linewidth=2, capthick=2)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, m + 0.003,
                f"{m:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    n_ppt = long[long["group"]==grp]["participant"].nunique()
    grp_label = "Single Target" if grp=="single" else "Multiple Target"
    ax.set_title(f"{grp_label} Group (n={n_ppt})", fontweight="bold")
    ax.set_ylabel("Mean Accuracy (proportion)")
    ax.set_ylim(0.85, 1.08)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1,
               alpha=0.5, label="Perfect accuracy (1.0)")
    ax.legend(fontsize=9)

lab_patch  = mpatches.Patch(color=C["lab"],  alpha=0.85, label="Lab Task")
game_patch = mpatches.Patch(color=C["game"], alpha=0.85, label="Game")
fig.legend(handles=[lab_patch, game_patch],
           loc="upper center", ncol=2, fontsize=11,
           bbox_to_anchor=(0.5, 1.0), framealpha=0.9)

fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig5_accuracy_barchart.png", bbox_inches="tight")
plt.close()
print("✅ Fig 5 saved — Accuracy bar chart")


# ─────────────────────────────────────────────────────────────
# FIG 6 — Line plot: RT across game levels (RQ4)
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
fig.suptitle("Figure 6: Mean Reaction Time Across Game Levels by Group (RQ4)",
             fontsize=15, fontweight="bold")

completed = levels[levels["attempt_status"] == "completed"]

_lvl_store = {}
for grp, col, label in [
    ("single",   C["single"],   "Single Target"),
    ("multiple", C["multiple"], "Multiple Target"),
]:
    sub      = completed[completed["group"] == grp]
    lvl_data = sub.groupby("level")["RT_ms"].agg(["mean","std","count"]).reset_index()
    lvl_data["se"] = lvl_data["std"] / np.sqrt(lvl_data["count"])
    lvl_data["ci"] = stats.t.ppf(0.975, df=lvl_data["count"]-1) * lvl_data["se"]
    _lvl_store[grp] = (lvl_data, col)

    ax.plot(lvl_data["level"], lvl_data["mean"],
            color=col, linewidth=2.5, marker="o",
            markersize=7, label=label, zorder=3)
    ax.fill_between(lvl_data["level"],
                    lvl_data["mean"] - lvl_data["ci"],
                    lvl_data["mean"] + lvl_data["ci"],
                    color=col, alpha=0.15)

# Separate y-rows per group so n-labels do not overlap
_y_base = min(d[0]["mean"].min() for d in _lvl_store.values())
_y_rows = {"single": _y_base - 200, "multiple": _y_base - 450}
for _grp, (_ld, _col) in _lvl_store.items():
    for _, row in _ld.iterrows():
        ax.annotate(f"n={int(row['count'])}",
                    xy=(row["level"], _y_rows[_grp]),
                    ha="center", fontsize=7, color=_col, alpha=0.8)

ax.set_xlabel("Game Level (1 = easiest)", fontsize=12)
ax.set_ylabel("Mean Reaction Time (ms)", fontsize=12)
ax.set_xticks(range(1, 16))
ax.legend(fontsize=11, framealpha=0.9)
ax.set_ylim(bottom=max(0, completed["RT_ms"].min() - 500))
ax.annotate("↑ Increasing difficulty",
            xy=(0.75, 0.88), xycoords="axes fraction",
            fontsize=10, color="gray",
            arrowprops=dict(arrowstyle="->", color="gray"),
            xytext=(0.55, 0.80))

fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig6_level_rt_lineplot.png", bbox_inches="tight")
plt.close()
print("✅ Fig 6 saved — Level RT line plot")


# ─────────────────────────────────────────────────────────────
# FIG 7 — Boxplot: RT by target colour (red vs white) per group
# BUG V2 FIX: aggregate to participant-level means before testing
# BUG V3 FIX: use paired t-test (ttest_rel), not independent
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
fig.suptitle("Figure 7: RT by Target Colour in Lab Task — participant-level means\n"
             "(Feature Search — Treisman & Gelade, 1980)",
             fontsize=14, fontweight="bold")

colour_palette = {"red": "#E63946", "white": "#457B9D"}

for ax, grp in zip(axes, ["single", "multiple"]):
    sub = trials[trials["group"] == grp]

    # ── BUG V2 + V3 FIX: participant-level means, then paired test ──
    part_means = (
        sub.groupby(["participant", "target_colour"])["RT_ms"]
        .mean()
        .reset_index()
        .pivot(index="participant", columns="target_colour", values="RT_ms")
        .dropna()
    )
    red_vals   = part_means["red"].values
    white_vals = part_means["white"].values
    n          = len(part_means)
    diff       = red_vals - white_vals

    # Normality of difference scores
    w_diff, p_diff = stats.shapiro(diff)
    diff_normal = p_diff > 0.05

    # Paired t-test (always computed for completeness)
    t_stat, p_t = stats.ttest_rel(red_vals, white_vals)
    d = diff.mean() / diff.std(ddof=1)

    # Wilcoxon if non-normal
    if not diff_normal:
        w_stat, p_w = stats.wilcoxon(red_vals, white_vals)

    # ── Plot at participant level (one dot = one participant mean) ──
    bp = ax.boxplot(
        [part_means["red"].values, part_means["white"].values],
        labels=["Red Target", "White Target"],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markerfacecolor="gray",
                        markersize=6, alpha=0.5),
        widths=0.45,
    )
    bp["boxes"][0].set_facecolor(colour_palette["red"])
    bp["boxes"][1].set_facecolor(colour_palette["white"])
    for patch in bp["boxes"]:
        patch.set_alpha(0.75)

    # Overlay individual participant means with connecting lines
    for red_m, white_m in zip(part_means["red"].values,
                               part_means["white"].values):
        ax.plot([1, 2], [red_m, white_m],
                color="gray", linewidth=0.6, alpha=0.4, zorder=2)

    jitter_r = np.random.normal(0, 0.05, n)
    jitter_w = np.random.normal(0, 0.05, n)
    ax.scatter(1 + jitter_r, part_means["red"].values,
               color=colour_palette["red"], edgecolors="black",
               linewidths=0.5, s=40, zorder=3, alpha=0.85)
    ax.scatter(2 + jitter_w, part_means["white"].values,
               color=colour_palette["white"], edgecolors="black",
               linewidths=0.5, s=40, zorder=3, alpha=0.85)

    # Annotation with correct test and correct level of analysis
    if diff_normal:
        sig_str = "p < .001" if p_t < 0.001 else (f"p = {p_t:.3f}" + (" *" if p_t < 0.05 else " (n.s.)"))
        annot = (f"Paired t({n-1}) = {t_stat:.2f}\n"
                 f"{sig_str}, d = {d:.2f}\n"
                 f"Shapiro (diff): p = {p_diff:.3f} ✓")
    else:
        sig_str_w = "p < .001" if p_w < 0.001 else (f"p = {p_w:.3f}" + (" *" if p_w < 0.05 else " (n.s.)"))
        annot = (f"Wilcoxon: {sig_str_w}\n"
                 f"(diff non-normal: p = {p_diff:.3f})\n"
                 f"Paired t({n-1}) = {t_stat:.2f} (for ref.)")

    ax.annotate(annot,
                xy=(0.5, 0.96), xycoords="axes fraction",
                ha="center", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.8))

    grp_label = "Single Target" if grp == "single" else "Multiple Target"
    ax.set_title(f"{grp_label} Group\n(n={n} participant means)",
                 fontweight="bold")
    ax.set_ylabel("Mean RT per participant (ms)")

fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig7_colour_rt_boxplot.png", bbox_inches="tight")
plt.close()
print("✅ Fig 7 saved — Target colour RT boxplot (participant-level, paired test)")


print(f"""
{'='*55}
  ALL FIGURES SAVED TO: {FIG_DIR}/
{'='*55}
  Fig 1 — fig1_rt_histograms.png
  Fig 2 — fig2_rt_boxplots.png
  Fig 3 — fig3_rt_barchart.png
  Fig 4 — fig4_validity_scatter.png   (Spearman/Pearson per group)
  Fig 5 — fig5_accuracy_barchart.png
  Fig 6 — fig6_level_rt_lineplot.png
  Fig 7 — fig7_colour_rt_boxplot.png  (participant-level, paired)
{'='*55}
""")
