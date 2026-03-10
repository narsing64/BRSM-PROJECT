"""
=============================================================
  Selective Attention Study — Descriptive Statistics
=============================================================

READS:
  processed_data/master_dataset_long.csv   (datasets 1)
  processed_data/master_dataset_wide.csv   (dataset 2)
  processed_data/game_levels_long.csv      (dataset 3)
  processed_data/lab_trials_long.csv       (dataset 4)

OUTPUTS (all saved to processed_data/):
  descriptive_summary.csv     — main 2×2 table (Mean, SD, SE, CI)
  normality_results.csv       — Shapiro-Wilk per cell
  outlier_report.csv          — IQR-based outlier flags
  level_descriptives.csv      — RT/accuracy per level (RQ4)
  trial_descriptives.csv      — RT per trial type (colour effect)

SECTIONS:
  1. Core 2×2 Descriptives   (Mean, SD, SE, 95%CI, Min, Max)
  2. Marginal Means           (per group, per modality)
  3. Normality Tests          (Shapiro-Wilk per cell)
  4. Outlier Detection        (IQR method)
  5. Distribution Shape       (Skewness, Kurtosis)
  6. Level-wise Descriptives  (RQ4)
  7. Trial-level Descriptives (colour effect, reliability prep)
=============================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

INPUT_DIR  = "processed_data"
OUTPUT_DIR = "processed_data"

# ── Load all datasets ──────────────────────────────────────
long   = pd.read_csv(f"{INPUT_DIR}/master_dataset_long.csv")
wide   = pd.read_csv(f"{INPUT_DIR}/master_dataset_wide.csv")
levels = pd.read_csv(f"{INPUT_DIR}/game_levels_long.csv")
trials = pd.read_csv(f"{INPUT_DIR}/lab_trials_long.csv")

print("=" * 60)
print("  SELECTIVE ATTENTION STUDY — DESCRIPTIVE STATISTICS")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# SECTION 1 — CORE 2×2 DESCRIPTIVES
# 4 cells: single/lab, single/game, multiple/lab, multiple/game
# Metrics: N, Mean, SD, SE, 95% CI lower/upper, Min, Max, Median
# ─────────────────────────────────────────────────────────────
print("\n── SECTION 1: Core 2×2 Descriptives ──────────────────")

def descriptives(series):
    """Compute full descriptive stats for a numeric series."""
    n    = len(series)
    mean = series.mean()
    sd   = series.std(ddof=1)
    se   = sd / np.sqrt(n)
    ci   = stats.t.ppf(0.975, df=n-1) * se   # 95% CI half-width
    return pd.Series({
        "N"       : n,
        "Mean"    : round(mean, 3),
        "SD"      : round(sd,   3),
        "SE"      : round(se,   3),
        "CI_lower": round(mean - ci, 3),
        "CI_upper": round(mean + ci, 3),
        "Median"  : round(series.median(), 3),
        "Min"     : round(series.min(), 3),
        "Max"     : round(series.max(), 3),
    })

# RT — 2×2
rt_desc = long.groupby(["group", "modality"])["RT_ms"].apply(descriptives).unstack()
print("\n  Reaction Time (ms):")
print(rt_desc.to_string())

# Accuracy — 2×2
acc_desc = long.groupby(["group", "modality"])["accuracy"].apply(descriptives).unstack()
print("\n  Accuracy (proportion):")
print(acc_desc.to_string())

# Save combined table
rt_desc_df  = long.groupby(["group","modality"])["RT_ms"].apply(descriptives).reset_index()
rt_desc_df.columns  = ["group","modality"] + list(rt_desc_df.columns[2:])
rt_desc_df.insert(2, "measure", "RT_ms")

acc_desc_df = long.groupby(["group","modality"])["accuracy"].apply(descriptives).reset_index()
acc_desc_df.columns = ["group","modality"] + list(acc_desc_df.columns[2:])
acc_desc_df.insert(2, "measure", "accuracy")

summary_df = pd.concat([rt_desc_df, acc_desc_df], ignore_index=True)
summary_df.to_csv(f"{OUTPUT_DIR}/descriptive_summary.csv", index=False)
print(f"\n  ✅ Saved → descriptive_summary.csv")


# ─────────────────────────────────────────────────────────────
# SECTION 2 — MARGINAL MEANS
# Collapse across one factor at a time
# ─────────────────────────────────────────────────────────────
print("\n── SECTION 2: Marginal Means ───────────────────────────")

print("\n  RT by GROUP (collapsed across modality):")
print(long.groupby("group")["RT_ms"].apply(descriptives).unstack().to_string())

print("\n  RT by MODALITY (collapsed across group):")
print(long.groupby("modality")["RT_ms"].apply(descriptives).unstack().to_string())

print("\n  Accuracy by GROUP:")
print(long.groupby("group")["accuracy"].apply(descriptives).unstack().to_string())

print("\n  Accuracy by MODALITY:")
print(long.groupby("modality")["accuracy"].apply(descriptives).unstack().to_string())


# ─────────────────────────────────────────────────────────────
# SECTION 3 — NORMALITY TESTS (Shapiro-Wilk)
# Required before choosing parametric vs non-parametric tests
# Null hypothesis: data is normally distributed
# p > 0.05 → do not reject normality
# ─────────────────────────────────────────────────────────────
print("\n── SECTION 3: Normality Tests (Shapiro-Wilk) ──────────")
print("  Interpretation: p > 0.05 = normal  |  p ≤ 0.05 = non-normal")

norm_rows = []
for (grp, mod), sub in long.groupby(["group", "modality"]):

    # RT normality
    w_rt, p_rt   = stats.shapiro(sub["RT_ms"])
    # Accuracy normality (skip if no variance — e.g. all 1.0)
    if sub["accuracy"].std() > 0:
        w_ac, p_ac = stats.shapiro(sub["accuracy"])
    else:
        w_ac, p_ac = np.nan, np.nan

    norm_rows.append({
        "group"         : grp,
        "modality"      : mod,
        "N"             : len(sub),
        "RT_W"          : round(w_rt, 4),
        "RT_p"          : round(p_rt, 4),
        "RT_normal"     : "YES" if p_rt > 0.05 else "NO",
        "Acc_W"         : round(w_ac, 4) if not np.isnan(w_ac) else "N/A (no variance)",
        "Acc_p"         : round(p_ac, 4) if not np.isnan(p_ac) else "N/A",
        "Acc_normal"    : "YES" if (not np.isnan(p_ac) and p_ac > 0.05) else ("NO" if not np.isnan(p_ac) else "N/A"),
    })

norm_df = pd.DataFrame(norm_rows)
print(norm_df.to_string(index=False))
norm_df.to_csv(f"{OUTPUT_DIR}/normality_results.csv", index=False)
print(f"\n  ✅ Saved → normality_results.csv")


# ─────────────────────────────────────────────────────────────
# SECTION 4 — OUTLIER DETECTION (IQR method)
# Outlier = value below Q1 - 1.5×IQR  OR  above Q3 + 1.5×IQR
# ─────────────────────────────────────────────────────────────
print("\n── SECTION 4: Outlier Detection (IQR Method) ──────────")

outlier_rows = []
for (grp, mod), sub in long.groupby(["group", "modality"]):
    for measure in ["RT_ms", "accuracy"]:
        col  = sub[measure]
        Q1   = col.quantile(0.25)
        Q3   = col.quantile(0.75)
        IQR  = Q3 - Q1
        lo   = Q1 - 1.5 * IQR
        hi   = Q3 + 1.5 * IQR
        outs = sub[(col < lo) | (col > hi)]

        for _, r in outs.iterrows():
            outlier_rows.append({
                "participant": r["participant"],
                "group"      : grp,
                "modality"   : mod,
                "measure"    : measure,
                "value"      : round(r[measure], 3),
                "lower_bound": round(lo, 3),
                "upper_bound": round(hi, 3),
                "direction"  : "HIGH" if r[measure] > hi else "LOW",
            })

        status = f"{len(outs)} outlier(s)" if len(outs) > 0 else "none"
        print(f"  {grp}/{mod}/{measure}: [{lo:.1f}, {hi:.1f}]  → {status}")

outlier_df = pd.DataFrame(outlier_rows)
if not outlier_df.empty:
    print(f"\n  Outlier details:")
    print(outlier_df.to_string(index=False))
else:
    print("\n  No outliers detected.")

outlier_df.to_csv(f"{OUTPUT_DIR}/outlier_report.csv", index=False)
print(f"\n  ✅ Saved → outlier_report.csv")


# ─────────────────────────────────────────────────────────────
# SECTION 5 — DISTRIBUTION SHAPE (Skewness & Kurtosis)
# Skewness: 0 = symmetric, >0 = right skew, <0 = left skew
# Kurtosis: 0 = normal (excess), >0 = heavy tails, <0 = light
# Rule of thumb: |skew| < 2 and |kurt| < 7 = acceptable
# ─────────────────────────────────────────────────────────────
print("\n── SECTION 5: Distribution Shape ──────────────────────")

shape_rows = []
for (grp, mod), sub in long.groupby(["group", "modality"]):
    sk_rt = stats.skew(sub["RT_ms"])
    kt_rt = stats.kurtosis(sub["RT_ms"])     # excess kurtosis
    sk_ac = stats.skew(sub["accuracy"])
    kt_ac = stats.kurtosis(sub["accuracy"])

    shape_rows.append({
        "group"     : grp,
        "modality"  : mod,
        "RT_skew"   : round(sk_rt, 3),
        "RT_kurt"   : round(kt_rt, 3),
        "RT_shape"  : "OK" if abs(sk_rt) < 2 and abs(kt_rt) < 7 else "CHECK",
        "Acc_skew"  : round(sk_ac, 3),
        "Acc_kurt"  : round(kt_ac, 3),
        "Acc_shape" : "OK" if abs(sk_ac) < 2 and abs(kt_ac) < 7 else "CHECK",
    })

shape_df = pd.DataFrame(shape_rows)
print(shape_df.to_string(index=False))


# ─────────────────────────────────────────────────────────────
# SECTION 6 — LEVEL-WISE DESCRIPTIVES  (RQ4)
# Mean RT and accuracy per level, separately for each group
# ─────────────────────────────────────────────────────────────
print("\n── SECTION 6: Level-wise Descriptives (Game — RQ4) ────")

# Only completed levels for clean analysis
completed = levels[levels["attempt_status"] == "completed"].copy()

level_desc_rows = []
for (grp, lvl), sub in completed.groupby(["group", "level"]):
    level_desc_rows.append({
        "group"       : grp,
        "level"       : lvl,
        "N"           : len(sub),
        "Mean_RT"     : round(sub["RT_ms"].mean(), 2),
        "SD_RT"       : round(sub["RT_ms"].std(ddof=1), 2),
        "Mean_HitRate": round(sub["hit_rate"].mean(), 4),
        "SD_HitRate"  : round(sub["hit_rate"].std(ddof=1), 4),
        "Mean_FA"     : round(sub["false_alarms"].mean(), 3),
    })

level_desc = pd.DataFrame(level_desc_rows)
level_desc.to_csv(f"{OUTPUT_DIR}/level_descriptives.csv", index=False)

print("\n  Single group — Mean RT per level:")
single_lvl = level_desc[level_desc["group"] == "single"][["level","N","Mean_RT","SD_RT","Mean_HitRate"]]
print(single_lvl.to_string(index=False))

print("\n  Multiple group — Mean RT per level:")
multi_lvl = level_desc[level_desc["group"] == "multiple"][["level","N","Mean_RT","SD_RT","Mean_HitRate"]]
print(multi_lvl.to_string(index=False))

print(f"\n  ✅ Saved → level_descriptives.csv")


# ─────────────────────────────────────────────────────────────
# SECTION 7 — TRIAL-LEVEL DESCRIPTIVES
# RT by target colour (red vs white) per group
# Within-person SD (individual variability)
# ─────────────────────────────────────────────────────────────
print("\n── SECTION 7: Trial-level Descriptives (Lab) ──────────")

# RT by colour × group
print("\n  Mean RT by target colour × group:")
colour_desc = trials.groupby(["group","target_colour"])["RT_ms"].apply(descriptives).unstack()
print(colour_desc.to_string())

# Within-person variability (SD of RT across 15 trials per participant)
within_sd = trials.groupby(["participant","group"])["RT_ms"].std(ddof=1).reset_index()
within_sd.columns = ["participant", "group", "within_person_SD"]

print("\n  Within-person RT variability (SD across 15 trials):")
print(within_sd.groupby("group")["within_person_SD"].apply(descriptives).unstack().to_string())

trial_desc_rows = []
for (grp, col), sub in trials.groupby(["group","target_colour"]):
    trial_desc_rows.append({
        "group"         : grp,
        "target_colour" : col,
        "N_trials"      : len(sub),
        "Mean_RT"       : round(sub["RT_ms"].mean(), 3),
        "SD_RT"         : round(sub["RT_ms"].std(ddof=1), 3),
        "Mean_accuracy" : round(sub["accuracy"].mean(), 4),
    })

trial_desc = pd.DataFrame(trial_desc_rows)
trial_desc.to_csv(f"{OUTPUT_DIR}/trial_descriptives.csv", index=False)
print(f"\n  ✅ Saved → trial_descriptives.csv")


# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY PRINTOUT
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SUMMARY FOR REPORT")
print("=" * 60)

print("""
  2×2 CELL MEANS (RT in ms, Accuracy as proportion)
  ┌─────────────┬──────────────────┬──────────────────┐
  │             │   Lab            │   Game           │
  ├─────────────┼──────────────────┼──────────────────┤""")

for grp in ["single", "multiple"]:
    lab_rt  = long[(long["group"]==grp)&(long["modality"]=="lab")]["RT_ms"].mean()
    lab_ac  = long[(long["group"]==grp)&(long["modality"]=="lab")]["accuracy"].mean()
    lab_rt_sd = long[(long["group"]==grp)&(long["modality"]=="lab")]["RT_ms"].std()
    gam_rt  = long[(long["group"]==grp)&(long["modality"]=="game")]["RT_ms"].mean()
    gam_ac  = long[(long["group"]==grp)&(long["modality"]=="game")]["accuracy"].mean()
    gam_rt_sd = long[(long["group"]==grp)&(long["modality"]=="game")]["RT_ms"].std()
    n       = long[long["group"]==grp]["participant"].nunique()
    print(f"  │ {grp:11s} │ RT={lab_rt:.0f}({lab_rt_sd:.0f})  │ RT={gam_rt:.0f}({gam_rt_sd:.0f})  │")
    print(f"  │ (n={n:2d})       │ Acc={lab_ac:.3f}        │ Acc={gam_ac:.3f}        │")
    print(f"  ├─────────────┼──────────────────┼──────────────────┤")

print("""  └─────────────┴──────────────────┴──────────────────┘
  Values shown as Mean(SD)
""")

norm_issues = norm_df[norm_df["RT_normal"] == "NO"]
if not norm_issues.empty:
    print("  ⚠  Non-normal RT distributions (consider non-parametric tests):")
    for _, r in norm_issues.iterrows():
        print(f"     {r['group']}/{r['modality']}: W={r['RT_W']}, p={r['RT_p']}")
else:
    print("  ✓  All RT distributions are approximately normal (p > 0.05)")

n_out = len(outlier_df)
if n_out > 0:
    print(f"\n  ⚠  {n_out} outlier(s) detected — see outlier_report.csv")
else:
    print("\n  ✓  No outliers detected")

print("\n✅  All descriptive statistics complete.")
print(f"    Files saved in: {OUTPUT_DIR}/")