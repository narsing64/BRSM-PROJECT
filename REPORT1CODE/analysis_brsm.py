"""
=============================================================
BUILD MASTER DATASET — Selective Attention Game Study
=============================================================
Author  : [Your Name]
Date    : March 2026

FOLDER STRUCTURE EXPECTED:
    DATA_BRSM/
    ├── single/
    │   ├── lab/      → 1_visual_search_*.csv  ...  21_visual_search_*.csv
    │   └── phone/    → 1_attentional_spotter_results.csv  ...  21_*.csv
    └── multiple/
        ├── lab/      → 22_visual_search_*.csv  ...  37_visual_search_*.csv
        └── phone/    → 22_attentional_spotter_results.csv  ...  37_*.csv

OUTPUT:
    master_dataset.csv          — one row per participant × modality (74 rows)
    master_wide.csv             — one row per participant (37 rows), wide format
    trial_level_dataset.csv     — every individual trial/round kept

INSTALL REQUIREMENTS:
    pip install pandas numpy scipy
=============================================================
"""

import os
import ast
import re
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# ❶  SET THIS TO YOUR DATA ROOT FOLDER
# ─────────────────────────────────────────────────────────────
DATA_ROOT = "."          # change if your folder is elsewhere
OUTPUT_DIR = "output_master"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# ❷  HELPER: Extract participant ID from filename
# ─────────────────────────────────────────────────────────────
def extract_pid(filename):
    """
    Extracts leading number from filename.
    '25_visual_search_2026-02-06_16h02.12.148.csv' → 25
    '1_attentional_spotter_results.csv'             → 1
    """
    match = re.match(r"^(\d+)", os.path.basename(filename))
    return int(match.group(1)) if match else None


# ─────────────────────────────────────────────────────────────
# ❸  PARSER: Single-Target LAB (PsychoPy CSV)
#     One click per trial → mouse.time is a single-element list
# ─────────────────────────────────────────────────────────────
def parse_single_lab(filepath):
    """
    Returns: list of dicts, one per trial.
    Key columns used:
        mouse.time          → RT (seconds, convert to ms)
        mouse.clicked_name  → accuracy check (should be 'target')
        target_col          → stimulus colour
        trials.thisN        → trial number
    """
    df = pd.read_csv(filepath, low_memory=False)
    pid = extract_pid(filepath)

    # Keep only data rows (have mouse.time filled)
    df = df[df["mouse.time"].notna()].copy()
    df = df[df["mouse.time"].astype(str).str.startswith("[")].copy()

    records = []
    for _, row in df.iterrows():
        try:
            time_list = ast.literal_eval(str(row["mouse.time"]))
            rt_ms     = float(time_list[0]) * 1000          # single target → first (only) click

            clicked   = str(row.get("mouse.clicked_name", ""))
            correct   = 1 if "target" in clicked else 0

            records.append({
                "ParticipantID"  : pid,
                "Group"          : "Single",
                "Modality"       : "Lab",
                "Trial"          : int(row.get("trials.thisN", len(records))),
                "TargetColour"   : str(row.get("target_col", "")),
                "RT_ms"          : rt_ms,
                "Correct"        : correct,
                "FalseAlarms"    : 0,         # lab task has no false alarm metric
                "NTargets"       : 1,
            })
        except Exception:
            continue   # skip malformed rows silently

    return records


# ─────────────────────────────────────────────────────────────
# ❹  PARSER: Multiple-Target LAB (PsychoPy CSV)
#     Five clicks per trial → mouse.time is a 5-element list
# ─────────────────────────────────────────────────────────────
def parse_multi_lab(filepath):
    """
    Returns: list of dicts, one per trial.
    RT_ms          = total trial time (time to last / 5th click) × 1000
    FirstClickRT   = time to first click × 1000
    AvgInterTarget = mean gap between successive clicks (ms)
    """
    df = pd.read_csv(filepath, low_memory=False)
    pid = extract_pid(filepath)

    df = df[df["mouse.time"].notna()].copy()
    df = df[df["mouse.time"].astype(str).str.startswith("[")].copy()

    records = []
    for _, row in df.iterrows():
        try:
            time_list = ast.literal_eval(str(row["mouse.time"]))
            time_list_ms = [t * 1000 for t in time_list]

            rt_ms          = time_list_ms[-1]              # total completion time
            first_click_ms = time_list_ms[0]              # first target detected
            n_clicks       = len(time_list_ms)

            # Inter-target intervals
            if n_clicks > 1:
                intervals = [time_list_ms[i+1] - time_list_ms[i]
                             for i in range(n_clicks - 1)]
                avg_inter = np.mean(intervals)
            else:
                avg_inter = 0

            clicked = str(row.get("mouse.clicked_name", ""))
            n_correct = clicked.count("target")

            records.append({
                "ParticipantID"  : pid,
                "Group"          : "Multi",
                "Modality"       : "Lab",
                "Trial"          : int(row.get("trials.thisN", len(records))),
                "TargetColour"   : str(row.get("target_col", "")),
                "RT_ms"          : rt_ms,              # total time for all 5 targets
                "FirstClickRT_ms": first_click_ms,     # comparable to Game InitialRT
                "AvgInterTarget_ms": avg_inter,
                "Correct"        : 1 if n_correct >= 5 else 0,
                "NCorrect"       : n_correct,
                "FalseAlarms"    : 0,
                "NTargets"       : 5,
            })
        except Exception:
            continue

    return records


# ─────────────────────────────────────────────────────────────
# ❺  PARSER: Single-Target GAME (attentional_spotter_results.csv)
#     One target per level → InitialResponseTime only
# ─────────────────────────────────────────────────────────────
def parse_single_game(filepath):
    """
    Key columns:
        Level                   → difficulty level
        InitialResponseTime(ms) → RT to single target
        SuccessRate(%)          → accuracy (should be 100 for single)
        FalseAlarms             → incorrect taps
        Completed               → whether level was finished
        HitPositions(x,y)       → spatial position (stored as string)
    """
    df = pd.read_csv(filepath, low_memory=False)
    pid = extract_pid(filepath)

    # Keep only singleTarget rows (safety check)
    if "GameMode" in df.columns:
        df = df[df["GameMode"].str.contains("single", case=False, na=False)]

    records = []
    for i, row in df.iterrows():
        try:
            records.append({
                "ParticipantID"   : pid,
                "Group"           : "Single",
                "Modality"        : "Game",
                "Trial"           : int(row.get("Level", i)),
                "Level"           : int(row.get("Level", i)),
                "RT_ms"           : float(row["InitialResponseTime(ms)"]),
                "FirstClickRT_ms" : float(row["InitialResponseTime(ms)"]),  # same for single
                "SuccessRate_pct" : float(row.get("SuccessRate(%)", 100)),
                "Correct"         : 1 if float(row.get("SuccessRate(%)", 100)) == 100 else 0,
                "FalseAlarms"     : int(row.get("FalseAlarms", 0)),
                "Completed"       : str(row.get("Completed", "true")).lower() == "true",
                "FinalScore"      : int(row.get("FinalScore", 0)),
                "NTargets"        : 1,
                "HitPositions"    : str(row.get("HitPositions(x,y)", "")),
            })
        except Exception:
            continue

    return records


# ─────────────────────────────────────────────────────────────
# ❻  PARSER: Multiple-Target GAME (attentional_spotter_results.csv)
#     Multiple targets per round → InitialRT + AvgInterTargetTime
# ─────────────────────────────────────────────────────────────
def parse_multi_game(filepath):
    """
    Key columns:
        Level                    → difficulty level
        InitialResponseTime(ms)  → time to FIRST target
        AvgInterTargetTime(ms)   → mean gap between successive taps
        SuccessRate(%)           → accuracy (% targets correctly found)
        HitRate(%)               → hit rate
        FalseAlarms              → incorrect taps
        Completed                → whether round was finished
    """
    df = pd.read_csv(filepath, low_memory=False)
    pid = extract_pid(filepath)

    if "GameMode" in df.columns:
        df = df[df["GameMode"].str.contains("multiple", case=False, na=False)]

    records = []
    for i, row in df.iterrows():
        try:
            init_rt   = float(row["InitialResponseTime(ms)"])
            inter_rt  = float(row.get("AvgInterTargetTime(ms)", 0))
            # Estimated total completion time: InitialRT + sum of inter-target gaps
            # We don't know n_targets exactly from this file alone, but AvgInterTarget × (n-1) approximates it
            success   = float(row.get("SuccessRate(%)", 100))

            records.append({
                "ParticipantID"     : pid,
                "Group"             : "Multi",
                "Modality"          : "Game",
                "Trial"             : i,
                "Level"             : int(row.get("Level", i)),
                "RT_ms"             : init_rt,               # InitialRT (first target)
                "FirstClickRT_ms"   : init_rt,
                "AvgInterTarget_ms" : inter_rt,
                "SuccessRate_pct"   : success,
                "HitRate_pct"       : float(row.get("HitRate(%)", 100)),
                "Correct"           : 1 if success == 100 else 0,
                "FalseAlarms"       : int(row.get("FalseAlarms", 0)),
                "Completed"         : str(row.get("Completed", "true")).lower() == "true",
                "FinalScore"        : int(row.get("FinalScore", 0)),
                "NTargets"          : None,   # not directly given per round
                "HitPositions"      : str(row.get("HitPositions(x,y)", "")),
            })
        except Exception:
            continue

    return records


# ─────────────────────────────────────────────────────────────
# ❼  AGGREGATE: Trial-level → Participant-level summary
# ─────────────────────────────────────────────────────────────
def aggregate_participant(trial_records):
    """
    Takes list of trial dicts for ONE participant × ONE modality,
    returns a single summary dict (one row in master dataset).
    """
    if not trial_records:
        return None

    df = pd.DataFrame(trial_records)
    pid      = df["ParticipantID"].iloc[0]
    group    = df["Group"].iloc[0]
    modality = df["Modality"].iloc[0]

    summary = {
        "ParticipantID"     : pid,
        "Group"             : group,
        "Modality"          : modality,
        "N_Trials"          : len(df),

        # ── Reaction Time ──────────────────────────────────
        "Mean_RT_ms"        : df["RT_ms"].mean(),
        "SD_RT_ms"          : df["RT_ms"].std(ddof=1),
        "Median_RT_ms"      : df["RT_ms"].median(),
        "Min_RT_ms"         : df["RT_ms"].min(),
        "Max_RT_ms"         : df["RT_ms"].max(),

        # ── Accuracy ───────────────────────────────────────
        "Mean_Accuracy"     : df["Correct"].mean(),
        "Total_FalseAlarms" : df["FalseAlarms"].sum(),
    }

    # Optional columns (not always present)
    if "FirstClickRT_ms" in df.columns:
        summary["Mean_FirstClickRT_ms"] = df["FirstClickRT_ms"].mean()
        summary["SD_FirstClickRT_ms"]   = df["FirstClickRT_ms"].std(ddof=1)

    if "AvgInterTarget_ms" in df.columns:
        vals = df["AvgInterTarget_ms"].replace(0, np.nan).dropna()
        summary["Mean_AvgInterTarget_ms"] = vals.mean() if len(vals) > 0 else np.nan

    if "SuccessRate_pct" in df.columns:
        summary["Mean_SuccessRate_pct"] = df["SuccessRate_pct"].mean()

    if "Completed" in df.columns:
        summary["N_Completed"]    = df["Completed"].sum()
        summary["Completion_Rate"] = df["Completed"].mean()

    if "Level" in df.columns:
        summary["Max_Level_Reached"] = df["Level"].max()

    if "FinalScore" in df.columns:
        summary["Total_Score"] = df["FinalScore"].sum()

    return summary


# ─────────────────────────────────────────────────────────────
# ❽  MAIN: Walk all folders, parse everything, build datasets
# ─────────────────────────────────────────────────────────────
def build_master_dataset(data_root):
    """
    Walks DATA_BRSM folder tree and builds:
      1. trial_records   — every trial row
      2. participant_rows — one row per participant × modality
    """
    trial_records    = []
    participant_rows = []

    # Map folder names to (group, modality, parser_function)
    folder_map = {
        os.path.join("single",   "lab")   : ("Single", "Lab",  parse_single_lab),
        os.path.join("single",   "phone") : ("Single", "Game", parse_single_game),
        os.path.join("multiple", "lab")   : ("Multi",  "Lab",  parse_multi_lab),
        os.path.join("multiple", "phone") : ("Multi",  "Game", parse_multi_game),
    }

    print("=" * 65)
    print("  BUILDING MASTER DATASET")
    print(f"  Data root: {os.path.abspath(data_root)}")
    print("=" * 65)

    for rel_path, (group, modality, parser_fn) in folder_map.items():
        folder = os.path.join(data_root, rel_path)

        if not os.path.isdir(folder):
            print(f"\n⚠  Folder NOT FOUND: {folder}")
            print(f"   Check your DATA_ROOT setting at the top of this script.")
            continue

        csv_files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")])
        print(f"\n📂 {rel_path}/ — {len(csv_files)} files found")

        for fname in csv_files:
            fpath = os.path.join(folder, fname)
            pid   = extract_pid(fname)

            try:
                trials = parser_fn(fpath)
                if not trials:
                    print(f"   ⚠ {fname} → no valid trials extracted")
                    continue

                trial_records.extend(trials)
                summary = aggregate_participant(trials)
                if summary:
                    participant_rows.append(summary)
                    print(f"   ✓ P{pid:>2} | {group:<6} | {modality:<4} | "
                          f"N={len(trials):>2} trials | "
                          f"M_RT={summary['Mean_RT_ms']:>7.0f} ms | "
                          f"Acc={summary['Mean_Accuracy']:.3f}")

            except Exception as e:
                print(f"   ✗ {fname} → ERROR: {e}")

    return trial_records, participant_rows


# ─────────────────────────────────────────────────────────────
# ❾  RESHAPE: Long → Wide format for ANOVA
# ─────────────────────────────────────────────────────────────
def make_wide_format(master_long):
    """
    Pivots master_long so each participant has ONE row,
    with separate columns for Lab/Game RT and Accuracy.

    Output columns:
        ParticipantID | Group | Mean_RT_Lab | Mean_RT_Game |
        Mean_Acc_Lab  | Mean_Acc_Game | RT_Diff | ...
    """
    lab  = master_long[master_long["Modality"] == "Lab"].copy()
    game = master_long[master_long["Modality"] == "Game"].copy()

    lab  = lab.add_suffix("_Lab").rename(columns={"ParticipantID_Lab": "ParticipantID",
                                                    "Group_Lab": "Group"})
    game = game.add_suffix("_Game").rename(columns={"ParticipantID_Game": "ParticipantID",
                                                     "Group_Game": "Group"})

    wide = pd.merge(lab, game, on=["ParticipantID", "Group"], how="outer")

    # Derived columns
    wide["RT_Diff_ms"]      = wide["Mean_RT_ms_Lab"]  - wide["Mean_RT_ms_Game"]
    wide["Acc_Diff"]        = wide["Mean_Accuracy_Lab"] - wide["Mean_Accuracy_Game"]

    # Numeric Group for ANOVA coding
    wide["Group_Code"] = wide["Group"].map({"Single": 0, "Multi": 1})

    return wide


# ─────────────────────────────────────────────────────────────
# ❿  VALIDITY CHECK: Print what we expect vs what we got
# ─────────────────────────────────────────────────────────────
def print_data_audit(master_long):
    print("\n" + "=" * 65)
    print("  DATA AUDIT")
    print("=" * 65)

    expected = {"Single": 21, "Multi": 16}
    for group, exp_n in expected.items():
        for mod in ["Lab", "Game"]:
            subset = master_long[(master_long["Group"] == group) &
                                  (master_long["Modality"] == mod)]
            n = len(subset)
            status = "✓" if n == exp_n else f"⚠ expected {exp_n}"
            print(f"  {group:<6} | {mod:<4} : {n:>2} participants  {status}")

    print(f"\n  Total rows in master_long : {len(master_long)}")
    print(f"  Expected                  : {(21+16)*2} = 74")


# ─────────────────────────────────────────────────────────────
# ⓫  DESCRIPTIVE STATS TABLE
# ─────────────────────────────────────────────────────────────
def print_descriptives(master_long):
    print("\n" + "=" * 65)
    print("  DESCRIPTIVE STATISTICS (participant-level means)")
    print("=" * 65)
    print(f"  {'Condition':<20} {'N':>4} {'M_RT (ms)':>12} {'SD_RT':>10} {'M_Acc':>8} {'SD_Acc':>8}")
    print("  " + "-" * 63)

    for group in ["Single", "Multi"]:
        for mod in ["Lab", "Game"]:
            sub = master_long[(master_long["Group"] == group) &
                               (master_long["Modality"] == mod)]
            if len(sub) == 0:
                continue
            label = f"{group}-{mod}"
            print(f"  {label:<20} "
                  f"{len(sub):>4} "
                  f"{sub['Mean_RT_ms'].mean():>12.1f} "
                  f"{sub['Mean_RT_ms'].std(ddof=1):>10.1f} "
                  f"{sub['Mean_Accuracy'].mean():>8.3f} "
                  f"{sub['Mean_Accuracy'].std(ddof=1):>8.3f}")


# ─────────────────────────────────────────────────────────────
# ⓬  STATISTICAL TESTS (preview — full in analysis script)
# ─────────────────────────────────────────────────────────────
def run_preview_stats(master_long, master_wide):
    print("\n" + "=" * 65)
    print("  PREVIEW STATISTICAL TESTS")
    print("=" * 65)

    # ── RQ1: Correlation (Game vs Lab RT) within each group ──
    print("\n  RQ1 — Concurrent Validity (Pearson's r)")
    for group in ["Single", "Multi"]:
        sub = master_wide[master_wide["Group"] == group].dropna(
            subset=["Mean_RT_ms_Lab", "Mean_RT_ms_Game"])
        if len(sub) < 3:
            print(f"    {group}: insufficient data (N={len(sub)})")
            continue
        r, p = stats.pearsonr(sub["Mean_RT_ms_Lab"], sub["Mean_RT_ms_Game"])
        sig  = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
        print(f"    {group:<6}: r = {r:+.3f} ({sig}), p = {p:.4f}, N = {len(sub)}")

    # ── RQ2: Target Load (Independent t-test, Lab RT) ──
    print("\n  RQ2 — Target Load Effect (Independent t-test, Lab RT)")
    lab_s = master_long[(master_long["Group"]=="Single") & (master_long["Modality"]=="Lab")]["Mean_RT_ms"]
    lab_m = master_long[(master_long["Group"]=="Multi")  & (master_long["Modality"]=="Lab")]["Mean_RT_ms"]
    if len(lab_s) > 1 and len(lab_m) > 1:
        t, p = stats.ttest_ind(lab_s, lab_m, equal_var=False)
        d    = (lab_m.mean() - lab_s.mean()) / np.sqrt((lab_s.std()**2 + lab_m.std()**2) / 2)
        sig  = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
        print(f"    Single M={lab_s.mean():.0f} vs Multi M={lab_m.mean():.0f}")
        print(f"    t = {t:.3f}, p = {p:.4f} ({sig}), Cohen's d = {d:.3f}")

    # ── RQ3: Modality Effect (Paired t-test) ──
    print("\n  RQ3 — Modality Effect (Paired t-test, within groups)")
    for group in ["Single", "Multi"]:
        sub = master_wide[master_wide["Group"] == group].dropna(
            subset=["Mean_RT_ms_Lab", "Mean_RT_ms_Game"])
        if len(sub) < 3:
            print(f"    {group}: insufficient data (N={len(sub)})")
            continue
        t, p = stats.ttest_rel(sub["Mean_RT_ms_Lab"], sub["Mean_RT_ms_Game"])
        d    = sub["RT_Diff_ms"].mean() / sub["RT_Diff_ms"].std(ddof=1)
        sig  = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
        print(f"    {group:<6}: t = {t:.3f}, p = {p:.4f} ({sig}), d = {d:.3f}, N = {len(sub)}")


# ─────────────────────────────────────────────────────────────
# ⓭  RUN EVERYTHING
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Step 1: Parse all files
    trial_records, participant_rows = build_master_dataset(DATA_ROOT)

    if not participant_rows:
        print("\n❌ No data was parsed. Check DATA_ROOT path and folder structure.")
        exit(1)

    # Step 2: Build long-format master dataset (74 rows when complete)
    master_long = pd.DataFrame(participant_rows)
    master_long = master_long.sort_values(["Group", "ParticipantID", "Modality"]).reset_index(drop=True)

    # Step 3: Build trial-level dataset (all raw trials)
    trial_df = pd.DataFrame(trial_records)
    trial_df = trial_df.sort_values(["Group", "ParticipantID", "Modality", "Trial"]).reset_index(drop=True)

    # Step 4: Build wide-format (37 rows, Lab + Game side by side)
    master_wide = make_wide_format(master_long)
    master_wide = master_wide.sort_values(["Group", "ParticipantID"]).reset_index(drop=True)

    # Step 5: Save all outputs
    long_path  = os.path.join(OUTPUT_DIR, "master_dataset_long.csv")
    wide_path  = os.path.join(OUTPUT_DIR, "master_dataset_wide.csv")
    trial_path = os.path.join(OUTPUT_DIR, "trial_level_dataset.csv")

    master_long.to_csv(long_path,  index=False)
    master_wide.to_csv(wide_path,  index=False)
    trial_df.to_csv(trial_path,    index=False)

    # Step 6: Print audit, descriptives, preview stats
    print_data_audit(master_long)
    print_descriptives(master_long)
    run_preview_stats(master_long, master_wide)

    print("\n" + "=" * 65)
    print("  OUTPUT FILES SAVED")
    print("=" * 65)
    print(f"  📄 {long_path}   — {len(master_long)} rows (participant × modality)")
    print(f"  📄 {wide_path}   — {len(master_wide)} rows (participant, wide)")
    print(f"  📄 {trial_path}  — {len(trial_df)} rows (every trial)")
    print("\n✅ Done. Load master_dataset_long.csv into SPSS or use the")
    print("   full analysis script for Mixed ANOVA, correlations, etc.")
    print("=" * 65)