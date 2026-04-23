"""
colour_effect_analysis.py
=========================
Corrected analysis for Figure 7 (Target Colour Effect).

WHAT THIS FIXES:
  The original analysis ran the t-test on raw trial-level data
  (555 observations), which is pseudoreplication — trials from the
  same participant are not independent. This inflates degrees of
  freedom and makes p-values too small.

CORRECT APPROACH:
  Step 1 — For each participant, compute their mean RT for red trials
            and their mean RT for white trials separately.
  Step 2 — Run a paired t-test on those participant-level means
            (21 pairs for single group, 16 pairs for multiple group).

HOW TO RUN:
  Set DATA_ROOT to your data folder (same as preprocess_final.py).
  Output: colour_effect_results.csv   — test statistics
          fig7_colour_rt_boxplot.png  — corrected figure
          (saved to OUTPUT_DIR)

UPLOAD colour_effect_results.csv back to Claude to update the report.
"""

import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── CONFIG — set these to match your machine ─────────────────────────
DATA_ROOT  = "."          # folder containing single/ and multiple/
OUTPUT_DIR = "processed_data"     # where to save outputs
# ─────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)


# ── 1. Load all lab trial files ───────────────────────────────────────

def load_lab_trials(group_folder, group_label):
    """
    Load all participant lab CSV files from a folder.
    Returns a DataFrame with columns:
        participant, group, target_col, RT_ms
    """
    rows = []
    lab_path = os.path.join(DATA_ROOT, group_folder, "lab")

    for fname in sorted(os.listdir(lab_path)):
        if not fname.endswith(".csv"):
            continue

        fpath = os.path.join(lab_path, fname)
        df = pd.read_csv(fpath)

        # Keep only valid trial rows
        trials = df[df["target_col"].notna()].copy()
        if len(trials) == 0:
            continue

        # Extract participant ID
        pid = trials["participant"].iloc[0] if "participant" in trials.columns else fname

        for _, row in trials.iterrows():
            try:
                times = ast.literal_eval(str(row["mouse.time"]))
                rt_ms = float(times[0]) * 1000      # first click, seconds -> ms
                colour = str(row["target_col"]).strip().lower()
                rows.append({
                    "participant": pid,
                    "group":       group_label,
                    "target_col":  colour,
                    "RT_ms":       rt_ms
                })
            except Exception:
                continue   # skip malformed rows

    return pd.DataFrame(rows)


print("Loading lab trial data...")
single_trials = load_lab_trials("single", "Single")
multi_trials  = load_lab_trials("multiple", "Multiple")
all_trials    = pd.concat([single_trials, multi_trials], ignore_index=True)

print(f"  Single group: {single_trials['participant'].nunique()} participants, "
      f"{len(single_trials)} trials")
print(f"  Multiple group: {multi_trials['participant'].nunique()} participants, "
      f"{len(multi_trials)} trials")
print()


# ── 2. Compute participant-level means per colour ─────────────────────

print("Computing participant-level colour means...")
part_means = (
    all_trials
    .groupby(["participant", "group", "target_col"])["RT_ms"]
    .mean()
    .reset_index()
    .rename(columns={"RT_ms": "mean_RT_ms"})
)

# Pivot so each participant has one row with red_mean and white_mean
part_wide = part_means.pivot_table(
    index=["participant", "group"],
    columns="target_col",
    values="mean_RT_ms"
).reset_index()

part_wide.columns.name = None
print(part_wide.head())
print()


# ── 3. Paired t-tests at participant level ─────────────────────────────

results = []

for group in ["Single", "Multiple"]:
    sub = part_wide[part_wide["group"] == group].copy()
    sub = sub.dropna(subset=["red", "white"])

    red_vals   = sub["red"].values
    white_vals = sub["white"].values
    n          = len(sub)

    t_stat, p_val = stats.ttest_rel(red_vals, white_vals)
    diff          = red_vals - white_vals
    d             = diff.mean() / diff.std(ddof=1)    # Cohen's d for paired

    # Normality of difference scores
    w_stat, w_p = stats.shapiro(diff)

    red_m,   red_sd   = red_vals.mean(),   red_vals.std(ddof=1)
    white_m, white_sd = white_vals.mean(), white_vals.std(ddof=1)

    print(f"=== {group} Target Group (n={n}) ===")
    print(f"  Red   M={red_m:.1f}ms  SD={red_sd:.1f}")
    print(f"  White M={white_m:.1f}ms  SD={white_sd:.1f}")
    print(f"  Paired t({n-1}) = {t_stat:.3f},  p = {p_val:.4f}")
    print(f"  Cohen's d = {d:.3f}")
    print(f"  Shapiro-Wilk (diff): W={w_stat:.3f}, p={w_p:.3f}")
    print(f"  Significant (p<.05): {p_val < 0.05}")
    print()

    results.append({
        "group":          group,
        "n":              n,
        "red_mean_RT":    round(red_m, 2),
        "red_sd_RT":      round(red_sd, 2),
        "white_mean_RT":  round(white_m, 2),
        "white_sd_RT":    round(white_sd, 2),
        "t_stat":         round(t_stat, 4),
        "df":             n - 1,
        "p_value":        round(p_val, 4),
        "cohens_d":       round(d, 4),
        "shapiro_W":      round(w_stat, 4),
        "shapiro_p":      round(w_p, 4),
        "significant":    p_val < 0.05
    })

results_df = pd.DataFrame(results)
out_csv = os.path.join(OUTPUT_DIR, "colour_effect_results.csv")
results_df.to_csv(out_csv, index=False)
print(f"Results saved to: {out_csv}")
print()


# ── 4. Regenerate Figure 7 using participant-level means ──────────────

print("Generating corrected Figure 7...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=False)
fig.suptitle("Mean RT by Target Colour (participant-level means)",
             fontsize=13, fontweight="bold", y=1.01)

colours_map  = {"red": "#e05c5c", "white": "#7a9fc4"}
display_cols = ["red", "white"]
labels       = ["Red", "White"]
groups       = ["Single", "Multiple"]

for ax, group in zip(axes, groups):
    sub = part_wide[part_wide["group"] == group].copy()
    sub = sub.dropna(subset=["red", "white"])

    # Boxplot data
    data = [sub["red"].values, sub["white"].values]
    bp = ax.boxplot(data, patch_artist=True, widths=0.45,
                    medianprops=dict(color="black", linewidth=2))

    for patch, col in zip(bp["boxes"], display_cols):
        patch.set_facecolor(colours_map[col])
        patch.set_alpha(0.7)

    # Overlay individual participant means
    for i, col in enumerate(display_cols, start=1):
        y = sub[col].values
        x = np.random.normal(i, 0.07, size=len(y))
        ax.scatter(x, y, color=colours_map[col], edgecolors="black",
                   linewidths=0.6, s=45, zorder=3, alpha=0.85)

    # Paired lines connecting each participant's red and white mean
    for _, row in sub.iterrows():
        ax.plot([1, 2], [row["red"], row["white"]],
                color="grey", linewidth=0.6, alpha=0.4, zorder=2)

    # Significance annotation
    res = results_df[results_df["group"] == group].iloc[0]
    p   = res["p_value"]
    t   = res["t_stat"]
    df  = res["df"]
    d   = res["cohens_d"]

    if p < 0.001:
        sig_str = "p < .001"
    elif p < 0.05:
        sig_str = f"p = {p:.3f}"
    else:
        sig_str = f"p = {p:.3f} (n.s.)"

    y_max = max(sub[["red", "white"]].max()) * 1.08
    ax.annotate(
        f"t({df}) = {t:.2f}\n{sig_str}\nd = {d:.2f}",
        xy=(1.5, y_max),
        ha="center", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                  edgecolor="grey", alpha=0.8)
    )

    ax.set_title(f"{group} Target Group (n={len(sub)})",
                 fontsize=11, fontweight="bold")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Red Target", "White Target"], fontsize=10)
    ax.set_ylabel("Mean RT (ms)", fontsize=10)
    ax.set_xlabel("Target Colour", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Legend
red_patch   = mpatches.Patch(color=colours_map["red"],   alpha=0.7, label="Red target")
white_patch = mpatches.Patch(color=colours_map["white"], alpha=0.7, label="White target")
fig.legend(handles=[red_patch, white_patch], loc="lower center",
           ncol=2, fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
out_fig = os.path.join(OUTPUT_DIR, "figures", "fig7_colour_rt_boxplot.png")
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close()
print(f"Figure saved to: {out_fig}")
print()
print("Done. Please upload colour_effect_results.csv to Claude.")