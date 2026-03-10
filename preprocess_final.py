"""
=============================================================
  Selective Attention Study — Final Preprocessing Script
=============================================================

GENERATES 4 DATASETS:

  1. master_dataset_long.csv  (74 rows)
     One row per participant × modality
     → Mixed ANOVA, independent t-test, paired t-test  [RQ2, RQ3]

  2. master_dataset_wide.csv  (37 rows)
     One row per participant, lab & game as side-by-side columns
     → Pearson correlation, scatter plots               [RQ1]

  3. game_levels_long.csv     (~500 rows)
     One row per participant × level (game only)
     → Repeated measures ANOVA on level effect          [RQ4]

  4. lab_trials_long.csv      (~555 rows)
     One row per participant × trial (lab only)
     → Reliability analysis (Cronbach's alpha)

BUGS FIXED vs original script:
  BUG 1 — Lab accuracy always 1.0
           single   → exact match: clicked[0] == "target"
           multiple → hits / N_TARGETS (5) per trial
  BUG 2 — Game used SuccessRate% (penalises FA) → fixed to HitRate%
  BUG 3 — Wrong comment "TOTAL search time" → "first response RT"

RETRY HANDLING (game levels):
  Participants could retry failed levels. Strategy:
  → Keep FIRST completed attempt per level per participant
  → If level never completed, keep last attempt + flag as "incomplete"
=============================================================
"""

import pandas as pd
import numpy as np
import os
import glob
import ast

# ─────────────────────────────────────────────────────────
# CONFIGURE PATHS
# ─────────────────────────────────────────────────────────
DATA_ROOT       = "."       # root folder containing single/ and multiple/
OUTPUT_DIR      = "processed_data"
N_TARGETS_MULTI = 5         # targets per trial in multiple condition

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────
def safe_parse(val):
    """Parse a stringified Python list from PsychoPy CSV cells."""
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return []


# ─────────────────────────────────────────────────────────
# ACCUMULATORS
# ─────────────────────────────────────────────────────────
master_rows     = []   # → dataset 1 & 2
lab_trial_rows  = []   # → dataset 4
level_rows      = []   # → dataset 3


# ─────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────
for group in ["single", "multiple"]:
    for modality in ["lab", "phone"]:

        folder = os.path.join(DATA_ROOT, group, modality)
        files  = sorted(glob.glob(os.path.join(folder, "*.csv")))

        if not files:
            print(f"  [WARNING] No CSV files in: {folder}")
            continue

        print(f"\n  Processing {group}/{modality} ({len(files)} files)...")

        for f in files:
            filename = os.path.basename(f)
            pid      = filename.split("_")[0]       # "22" from "22_visual_search_..."
            df       = pd.read_csv(f)

            # ─────────────────────────────────────
            # LAB FILES  →  datasets 1, 2, 4
            # ─────────────────────────────────────
            if modality == "lab":

                trial_df = df[df["target_col"].notna()].copy().reset_index(drop=True)

                rt_values  = []
                acc_values = []

                for _, row in trial_df.iterrows():
                    times   = safe_parse(row["mouse.time"])
                    clicked = safe_parse(row["mouse.clicked_name"])

                    if not (isinstance(times,   list) and len(times)   > 0):
                        continue
                    if not (isinstance(clicked, list) and len(clicked) > 0):
                        continue

                    # ── first response RT (ms)  [FIX BUG 3: was labelled "TOTAL"] ──
                    rt_ms = float(times[0]) * 1000

                    # ── accuracy per trial  [FIX BUG 1] ──
                    if group == "single":
                        # exact name match — name is literally 'target'
                        correct = 1 if clicked[0] == "target" else 0
                        acc     = float(correct)

                        # DATASET 4 row — single
                        lab_trial_rows.append({
                            "participant"  : pid,
                            "group"        : "single",
                            "trial_n"      : int(row["trials.thisN"]) if pd.notna(row["trials.thisN"]) else np.nan,
                            "target_colour": row["target_col"],
                            "RT_ms"        : rt_ms,
                            "correct"      : correct,
                            "n_hits"       : correct,
                            "accuracy"     : acc,
                        })

                    elif group == "multiple":
                        # count targets correctly clicked out of N_TARGETS_MULTI
                        
                        unique_hits = len(set(c for c in clicked if str(c).startswith("target_")))
                        acc  = unique_hits / N_TARGETS_MULTI

                        # DATASET 4 row — multiple
                        lab_trial_rows.append({
                            "participant"  : pid,
                            "group"        : "multiple",
                            "trial_n"      : int(row["trials.thisN"]) if pd.notna(row["trials.thisN"]) else np.nan,
                            "target_colour": row["target_col"],
                            "RT_ms"        : rt_ms,
                            "correct"      : 1 if unique_hits == N_TARGETS_MULTI else 0,
                            "n_hits"       : unique_hits,
                            "accuracy"     : acc,
                        })

                    rt_values.append(rt_ms)
                    acc_values.append(acc)

                # DATASET 1 row
                master_rows.append({
                    "participant": pid,
                    "group"      : group,
                    "modality"   : "lab",
                    "RT_ms"      : np.mean(rt_values)  if rt_values  else np.nan,
                    "accuracy"   : np.mean(acc_values) if acc_values else np.nan,
                })

            # ─────────────────────────────────────
            # GAME / PHONE FILES  →  datasets 1, 2, 3
            # ─────────────────────────────────────
            else:
                rt  = pd.to_numeric(df["InitialResponseTime(ms)"], errors="coerce")
                acc = pd.to_numeric(df["HitRate(%)"], errors="coerce") / 100   # FIX BUG 2

                # DATASET 1 row
                master_rows.append({
                    "participant": pid,
                    "group"      : group,
                    "modality"   : "game",
                    "RT_ms"      : rt.mean(),
                    "accuracy"   : acc.mean(),
                })

                # ── DATASET 3: level-wise rows ────────────────────────────────
                #
                # Retry handling:
                #   Keep FIRST completed attempt per level.
                #   If never completed, keep last attempt + flag "incomplete".
                #
                df["Completed"] = df["Completed"].astype(str).str.strip().str.lower()
                completed_df    = df[df["Completed"] == "true"].copy()

                if not completed_df.empty:
                    first_done = completed_df.groupby("Level", as_index=False).first()
                    first_done["attempt_status"] = "completed"
                else:
                    first_done = pd.DataFrame()

                done_levels  = set(first_done["Level"]) if not first_done.empty else set()
                never_done   = df[~df["Level"].isin(done_levels)].copy()

                if not never_done.empty:
                    last_fail = never_done.groupby("Level", as_index=False).last()
                    last_fail["attempt_status"] = "incomplete"
                else:
                    last_fail = pd.DataFrame()

                level_df = pd.concat(
                    [x for x in [first_done, last_fail] if not x.empty],
                    ignore_index=True
                )

                for _, lrow in level_df.iterrows():
                    level_rows.append({
                        "participant"        : pid,
                        "group"              : group,
                        "level"              : int(lrow["Level"]),
                        "attempt_status"     : lrow["attempt_status"],
                        "RT_ms"              : float(lrow["InitialResponseTime(ms)"]),
                        "hit_rate"           : float(lrow["HitRate(%)"]) / 100,
                        "success_rate"       : float(lrow["SuccessRate(%)"]) / 100,
                        "false_alarms"       : int(lrow["FalseAlarms"]),
                        "avg_inter_target_ms": float(lrow["AvgInterTargetTime(ms)"]) if "AvgInterTargetTime(ms)" in lrow else np.nan,
                        "final_score"        : float(lrow["FinalScore"]),
                    })


# ─────────────────────────────────────────────────────────
# DATASET 1 — master_dataset_long.csv
# ─────────────────────────────────────────────────────────
master_long = pd.DataFrame(master_rows)
master_long["participant"] = pd.to_numeric(master_long["participant"], errors="coerce")
master_long = master_long.sort_values(["participant", "modality"]).reset_index(drop=True)

out1 = os.path.join(OUTPUT_DIR, "master_dataset_long.csv")
master_long.to_csv(out1, index=False)
print(f"\n✅  Dataset 1 saved → {out1}  ({len(master_long)} rows)")


# ─────────────────────────────────────────────────────────
# DATASET 2 — master_dataset_wide.csv
# ─────────────────────────────────────────────────────────
wide = master_long.pivot_table(
    index   = ["participant", "group"],
    columns = "modality",
    values  = ["RT_ms", "accuracy"]
).reset_index()

# Flatten multi-level column names
wide.columns = [
    "participant", "group",
    "game_RT", "lab_RT",
    "game_acc", "lab_acc"
]
wide = wide.sort_values(["group", "participant"]).reset_index(drop=True)

out2 = os.path.join(OUTPUT_DIR, "master_dataset_wide.csv")
wide.to_csv(out2, index=False)
print(f"✅  Dataset 2 saved → {out2}  ({len(wide)} rows)")


# ─────────────────────────────────────────────────────────
# DATASET 3 — game_levels_long.csv
# ─────────────────────────────────────────────────────────
levels = pd.DataFrame(level_rows)
if not levels.empty:
    levels["participant"] = pd.to_numeric(levels["participant"], errors="coerce")
    levels = levels.sort_values(["group", "participant", "level"]).reset_index(drop=True)

out3 = os.path.join(OUTPUT_DIR, "game_levels_long.csv")
levels.to_csv(out3, index=False)
print(f"✅  Dataset 3 saved → {out3}  ({len(levels)} rows)")


# ─────────────────────────────────────────────────────────
# DATASET 4 — lab_trials_long.csv
# ─────────────────────────────────────────────────────────
lab_trials = pd.DataFrame(lab_trial_rows)
if not lab_trials.empty:
    lab_trials["participant"] = pd.to_numeric(lab_trials["participant"], errors="coerce")
    lab_trials = lab_trials.sort_values(["group", "participant", "trial_n"]).reset_index(drop=True)

out4 = os.path.join(OUTPUT_DIR, "lab_trials_long.csv")
lab_trials.to_csv(out4, index=False)
print(f"✅  Dataset 4 saved → {out4}  ({len(lab_trials)} rows)")


# ─────────────────────────────────────────────────────────
# VERIFICATION REPORT
# ─────────────────────────────────────────────────────────
sep = "\n" + "─" * 55

print(sep)
print("  DATASET 1 — master_dataset_long")
print("─" * 55)
print(master_long.groupby(["group", "modality"]).agg(
    N        = ("RT_ms",    "count"),
    Mean_RT  = ("RT_ms",    lambda x: round(x.mean(), 1)),
    SD_RT    = ("RT_ms",    lambda x: round(x.std(),  1)),
    Mean_Acc = ("accuracy", lambda x: round(x.mean(), 4)),
    SD_Acc   = ("accuracy", lambda x: round(x.std(),  4)),
).to_string())

print(sep)
print("  DATASET 2 — master_dataset_wide")
print("─" * 55)
print(wide.groupby("group").agg(
    N          = ("participant", "count"),
    Mean_lab_RT= ("lab_RT",  lambda x: round(x.mean(), 1)),
    Mean_gam_RT= ("game_RT", lambda x: round(x.mean(), 1)),
    Mean_lab_Ac= ("lab_acc", lambda x: round(x.mean(), 4)),
    Mean_gam_Ac= ("game_acc",lambda x: round(x.mean(), 4)),
).to_string())

print(sep)
print("  DATASET 3 — game_levels_long")
print("─" * 55)
print(f"  Total rows     : {len(levels)}")
print(f"  Participants   : {levels['participant'].nunique()}")
print(f"  Level range    : {levels['level'].min()} – {levels['level'].max()}")
print("\n  Attempt status :")
print(levels.groupby(["group", "attempt_status"]).size().to_string())
print("\n  Mean RT × group × level (levels 1–5) :")
print(levels[levels["level"] <= 5]
      .groupby(["group", "level"])["RT_ms"]
      .mean().round(1).to_string())

print(sep)
print("  DATASET 4 — lab_trials_long")
print("─" * 55)
print(f"  Total rows     : {len(lab_trials)}")
print(f"  Participants   : {lab_trials['participant'].nunique()}")
print(f"  Trials per ppt : {lab_trials.groupby('participant').size().unique()}")
print("\n  Accuracy range check (should NOT all be 1.0) :")
print(lab_trials.groupby(["group"])["accuracy"].agg(["min","max","mean"]).round(4).to_string())
print("\n  Target colour distribution :")
print(lab_trials.groupby(["group", "target_colour"]).size().to_string())

print("\n✅  All 4 datasets generated successfully.")