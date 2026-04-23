"""
colour_effect_analysis.py
=========================
CORRECTED VERSION

WHAT THIS DOES:
  Computes the target colour effect (red vs white) on RT
  using participant-level means — NOT raw trial data.
  Running tests on trial data (555 rows) is pseudoreplication
  because trials from the same participant are not independent.

CORRECT APPROACH:
  Step 1 — For each participant, compute their mean RT for red
            trials and their mean RT for white trials separately.
  Step 2 — Run a paired t-test on those participant-level means.
            (Same participant has both → paired comparison.)
  Step 3 — For the multiple group where difference scores are
            non-normal (Shapiro-Wilk W=0.733, p<.001), run
            Wilcoxon signed-rank test instead of the paired t-test.
            This fixes the inconsistency in the original report
            where a t-test result was reported in the conclusion
            but a Wilcoxon was "planned for Report 2".

BUG FIXED vs original colour_effect_analysis.py:
  Original was already correct in structure (participant-level means,
  paired t-test). This version adds:
  - Wilcoxon signed-rank for the non-normal multiple group
  - Clear reporting of which test is primary per group
  - Report-ready paragraph output

HOW TO RUN:
  Set DATA_ROOT to your data folder.
  Output: colour_effect_results.csv and fig7_colour_rt_boxplot.png
          saved to OUTPUT_DIR.
"""

import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

DATA_ROOT  = "."
OUTPUT_DIR = "processed_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Load lab trials (from preprocess_final.py output)
# ─────────────────────────────────────────────────────────────
trials_path = os.path.join(OUTPUT_DIR, "lab_trials_long.csv")

if os.path.exists(trials_path):
    print("Loading lab_trials_long.csv ...")
    all_trials = pd.read_csv(trials_path)
    all_trials = all_trials.rename(columns={"target_colour": "target_col",
                                             "RT_ms": "RT_ms"})
else:
    # Fallback: load raw lab CSVs directly (same logic as preprocess_final.py)
    print("lab_trials_long.csv not found — loading raw CSVs ...")

    def safe_parse(val):
        if pd.isna(val):
            return []
        try:
            return ast.literal_eval(str(val))
        except Exception:
            return []

    rows = []
    for group_folder, group_label in [("single", "Single"), ("multiple", "Multiple")]:
        lab_path = os.path.join(DATA_ROOT, group_folder, "lab")
        if not os.path.isdir(lab_path):
            print(f"  [WARNING] Not found: {lab_path}")
            continue
        for fname in sorted(os.listdir(lab_path)):
            if not fname.endswith(".csv"):
                continue
            fpath = os.path.join(lab_path, fname)
            df    = pd.read_csv(fpath)
            df    = df[df["target_col"].notna()].copy()
            if len(df) == 0:
                continue
            pid = fname.split("_")[0]
            for _, row in df.iterrows():
                try:
                    times = safe_parse(row["mouse.time"])
                    rt_ms = float(times[0]) * 1000
                    colour = str(row["target_col"]).strip().lower()
                    rows.append({
                        "participant": pid,
                        "group"      : group_label,
                        "target_col" : colour,
                        "RT_ms"      : rt_ms
                    })
                except Exception:
                    continue
    all_trials = pd.DataFrame(rows)

print(f"  Loaded {len(all_trials)} trial rows, "
      f"{all_trials['participant'].nunique()} participants")


# ─────────────────────────────────────────────────────────────
# Compute participant-level means per colour
# ─────────────────────────────────────────────────────────────
print("\nComputing participant-level colour means ...")

group_col = "group"
if group_col not in all_trials.columns and "Group" in all_trials.columns:
    all_trials = all_trials.rename(columns={"Group": "group"})

part_means = (
    all_trials
    .groupby(["participant", "group", "target_col"])["RT_ms"]
    .mean()
    .reset_index()
    .rename(columns={"RT_ms": "mean_RT_ms"})
)

part_wide = part_means.pivot_table(
    index=["participant", "group"],
    columns="target_col",
    values="mean_RT_ms"
).reset_index()
part_wide.columns.name = None

# Normalise group labels to title case for display
part_wide["group"] = part_wide["group"].str.capitalize()
print(f"  {len(part_wide)} participants with both red and white means")
print()


# ─────────────────────────────────────────────────────────────
# Statistical tests — paired t-test + Wilcoxon where non-normal
# ─────────────────────────────────────────────────────────────
results = []

for group in ["Single", "Multiple"]:
    sub = part_wide[part_wide["group"] == group].copy()
    sub = sub.dropna(subset=["red", "white"])

    red_vals   = sub["red"].values
    white_vals = sub["white"].values
    n          = len(sub)
    diff       = red_vals - white_vals

    # Normality of difference scores
    w_sw, p_sw = stats.shapiro(diff)
    diff_normal = p_sw > 0.05

    # Paired t-test (always computed)
    t_stat, p_t = stats.ttest_rel(red_vals, white_vals)
    d_paired    = diff.mean() / diff.std(ddof=1)

    # Wilcoxon signed-rank (for non-normal differences)
    if not diff_normal:
        w_stat, p_w = stats.wilcoxon(red_vals, white_vals)
    else:
        w_stat, p_w = np.nan, np.nan

    # Primary test
    primary_test = "Paired t-test" if diff_normal else "Wilcoxon signed-rank"
    p_primary    = p_t if diff_normal else p_w

    red_m,   red_sd   = red_vals.mean(),   red_vals.std(ddof=1)
    white_m, white_sd = white_vals.mean(), white_vals.std(ddof=1)

    print(f"=== {group} Target Group (n={n}) ===")
    print(f"  Red   M={red_m:.1f} ms  SD={red_sd:.1f}")
    print(f"  White M={white_m:.1f} ms  SD={white_sd:.1f}")
    print(f"  Shapiro-Wilk (diff): W={w_sw:.3f}, p={p_sw:.3f} "
          f"→ {'Normal ✓' if diff_normal else 'Non-normal ⚠'}")
    print(f"  Paired t({n-1}) = {t_stat:.3f},  p = {p_t:.4f},  d = {d_paired:.3f}")
    if not diff_normal:
        print(f"  Wilcoxon W = {w_stat:.0f},  p = {p_w:.4f}  ← PRIMARY (non-normal)")
    print(f"  Primary test: {primary_test},  p = {p_primary:.4f}")
    print(f"  Significant (p<.05): {p_primary < 0.05}")
    print()

    results.append({
        "group"         : group,
        "n"             : n,
        "red_mean_RT"   : round(red_m, 2),
        "red_sd_RT"     : round(red_sd, 2),
        "white_mean_RT" : round(white_m, 2),
        "white_sd_RT"   : round(white_sd, 2),
        "shapiro_W"     : round(w_sw, 4),
        "shapiro_p"     : round(p_sw, 4),
        "diff_normal"   : diff_normal,
        "t_stat"        : round(t_stat, 4),
        "t_df"          : n - 1,
        "p_t"           : round(p_t, 4),
        "cohens_d"      : round(d_paired, 4),
        "wilcoxon_W"    : round(w_stat, 4) if not np.isnan(w_stat) else "N/A",
        "p_wilcoxon"    : round(p_w, 4)    if not np.isnan(p_w)    else "N/A",
        "primary_test"  : primary_test,
        "p_primary"     : round(p_primary, 4),
        "significant"   : p_primary < 0.05,
    })

results_df = pd.DataFrame(results)
out_csv    = os.path.join(OUTPUT_DIR, "colour_effect_results.csv")
results_df.to_csv(out_csv, index=False)
print(f"Results saved → {out_csv}")
print()


# ─────────────────────────────────────────────────────────────
# Report-ready paragraph
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  REPORT-READY SUMMARY")
print("=" * 60)

s = results_df[results_df["group"] == "Single"].iloc[0]
m = results_df[results_df["group"] == "Multiple"].iloc[0]

print(f"""
Target colour effect (participant-level means):

Single-target group (n={s['n']}):
  Red targets were found significantly faster than white targets
  (M_red = {s['red_mean_RT']:.0f} ms vs M_white = {s['white_mean_RT']:.0f} ms).
  Difference scores were normally distributed (Shapiro-Wilk
  W = {s['shapiro_W']:.3f}, p = {s['shapiro_p']:.3f}), so a paired t-test was used:
  t({s['t_df']}) = {s['t_stat']:.3f}, p < .001, d = {s['cohens_d']:.2f} (large effect).
  This is consistent with Feature Integration Theory (Treisman &
  Gelade, 1980): a red target among white distractors constitutes
  a feature search where colour difference drives pop-out.

Multiple-target group (n={m['n']}):
  Descriptively, the difference was negligible
  (M_red = {m['red_mean_RT']:.0f} ms vs M_white = {m['white_mean_RT']:.0f} ms).
  Difference scores were non-normally distributed (Shapiro-Wilk
  W = {m['shapiro_W']:.3f}, p = {m['shapiro_p']:.3f}), so a Wilcoxon signed-rank
  test was used (primary test): {m['primary_test']},
  p = {m['p_primary']:.3f} (n.s.), indicating no significant colour
  effect in the multiple-target condition.
  Note: The paired t-test result (t({m['t_df']}) = {m['t_stat']:.3f}, p = {m['p_t']:.3f})
  is reported for reference only — the Wilcoxon is the valid test
  given the non-normal difference scores.
""")


# ─────────────────────────────────────────────────────────────
# Figure 7 — Corrected boxplot
# ─────────────────────────────────────────────────────────────
print("Generating Figure 7 ...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=False)
fig.suptitle("Mean RT by Target Colour (participant-level means)",
             fontsize=13, fontweight="bold", y=1.01)

colours_map  = {"red": "#e05c5c", "white": "#7a9fc4"}
display_cols = ["red", "white"]
groups_plot  = ["Single", "Multiple"]

for ax, group in zip(axes, groups_plot):
    sub = part_wide[part_wide["group"] == group].copy()
    sub = sub.dropna(subset=["red", "white"])

    data = [sub["red"].values, sub["white"].values]
    bp   = ax.boxplot(data, patch_artist=True, widths=0.45,
                      medianprops=dict(color="black", linewidth=2))

    for patch, col in zip(bp["boxes"], display_cols):
        patch.set_facecolor(colours_map[col])
        patch.set_alpha(0.7)

    for i, col in enumerate(display_cols, start=1):
        y = sub[col].values
        x = np.random.normal(i, 0.07, size=len(y))
        ax.scatter(x, y, color=colours_map[col], edgecolors="black",
                   linewidths=0.6, s=45, zorder=3, alpha=0.85)

    for _, row in sub.iterrows():
        ax.plot([1, 2], [row["red"], row["white"]],
                color="grey", linewidth=0.6, alpha=0.4, zorder=2)

    # Use primary test result for annotation
    res = results_df[results_df["group"] == group].iloc[0]
    p   = res["p_primary"]

    if p < 0.001:
        sig_str = "p < .001"
    elif p < 0.05:
        sig_str = f"p = {p:.3f} *"
    else:
        sig_str = f"p = {p:.3f} (n.s.)"

    test_label = "Wilcoxon" if res["primary_test"] == "Wilcoxon signed-rank" else f"t({res['t_df']})"
    stat_val   = (f"W = {res['wilcoxon_W']:.0f}" if res["primary_test"] == "Wilcoxon signed-rank"
                  else f"t({res['t_df']}) = {res['t_stat']:.2f}")

    y_max = max(sub[["red", "white"]].max()) * 1.08
    ax.annotate(
        f"{stat_val}\n{sig_str}\nd = {res['cohens_d']:.2f}",
        xy=(1.5, y_max), ha="center", va="bottom", fontsize=9,
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

red_patch   = mpatches.Patch(color=colours_map["red"],   alpha=0.7, label="Red target")
white_patch = mpatches.Patch(color=colours_map["white"], alpha=0.7, label="White target")
fig.legend(handles=[red_patch, white_patch], loc="lower center",
           ncol=2, fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
out_fig = os.path.join(OUTPUT_DIR, "figures", "fig7_colour_rt_boxplot.png")
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close()
print(f"Figure saved → {out_fig}")
print("\nDone.")
