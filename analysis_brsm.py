"""
=============================================================
  Selective Attention Study — Master Dataset Builder
  (CORRECTED VERSION)
=============================================================
BUGS FIXED vs original:
  BUG V5 — parse_multi_lab set RT_ms = time_list_ms[-1]
            (the LAST click = total completion time).
            The study protocol defines RT as InitialResponseTime
            = time to the FIRST click. This must match the game
            metric (InitialResponseTime(ms)) for valid comparison.
            FIX: RT_ms = time_list_ms[0] (first click).
            FirstClickRT_ms kept as alias, TotalRT_ms added
            separately for optional completeness analysis.

  BUG V6 — parse_multi_lab accuracy used
            clicked.count("target") which counts the substring
            "target" in the full string representation. This
            works by coincidence for "target_N" names but is
            fragile. FIX: parse the list properly and count
            unique items starting with "target_".

FOLDER STRUCTURE EXPECTED:
    DATA_BRSM/
    ├── single/
    │   ├── lab/      → 1_visual_search_*.csv ... 21_*.csv
    │   └── phone/    → 1_attentional_spotter_results.csv ...
    └── multiple/
        ├── lab/      → 22_visual_search_*.csv ... 37_*.csv
        └── phone/    → 22_attentional_spotter_results.csv ...

OUTPUT (saved to output_master/):
    master_dataset_long.csv   — 74 rows (participant × modality)
    master_dataset_wide.csv   — 37 rows (participant, wide)
    trial_level_dataset.csv   — every individual trial/level
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

DATA_ROOT  = "."
OUTPUT_DIR = "output_master"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_TARGETS_MULTI = 5   # targets per trial in the multiple condition


# ─────────────────────────────────────────────────────────────
# HELPER: Extract participant ID from filename
# ─────────────────────────────────────────────────────────────
def extract_pid(filename):
    match = re.match(r"^(\d+)", os.path.basename(filename))
    return int(match.group(1)) if match else None


# ─────────────────────────────────────────────────────────────
# HELPER: Safe parse of stringified Python list
# ─────────────────────────────────────────────────────────────
def safe_parse_list(val):
    if pd.isna(val):
        return []
    try:
        result = ast.literal_eval(str(val))
        return result if isinstance(result, list) else []
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────
# PARSER: Single-Target LAB
# ─────────────────────────────────────────────────────────────
def parse_single_lab(filepath):
    df  = pd.read_csv(filepath, low_memory=False)
    pid = extract_pid(filepath)

    df = df[df["mouse.time"].notna()].copy()
    df = df[df["mouse.time"].astype(str).str.startswith("[")].copy()

    records = []
    for _, row in df.iterrows():
        try:
            time_list = safe_parse_list(row["mouse.time"])
            if not time_list:
                continue
            rt_ms = float(time_list[0]) * 1000   # first (only) click → ms

            clicked = safe_parse_list(row.get("mouse.clicked_name", "[]"))
            correct = 1 if clicked and clicked[0] == "target" else 0

            records.append({
                "ParticipantID" : pid,
                "Group"         : "Single",
                "Modality"      : "Lab",
                "Trial"         : int(row.get("trials.thisN", len(records))),
                "TargetColour"  : str(row.get("target_col", "")),
                "RT_ms"         : rt_ms,
                "Correct"       : correct,
                "FalseAlarms"   : 0,
                "NTargets"      : 1,
            })
        except Exception:
            continue

    return records


# ─────────────────────────────────────────────────────────────
# PARSER: Multiple-Target LAB
# BUG V5 FIX: RT_ms = first click (not last)
# BUG V6 FIX: accuracy via set of target_ items (not substring count)
# ─────────────────────────────────────────────────────────────
def parse_multi_lab(filepath):
    df  = pd.read_csv(filepath, low_memory=False)
    pid = extract_pid(filepath)

    df = df[df["mouse.time"].notna()].copy()
    df = df[df["mouse.time"].astype(str).str.startswith("[")].copy()

    records = []
    for _, row in df.iterrows():
        try:
            time_list    = safe_parse_list(row["mouse.time"])
            time_list_ms = [t * 1000 for t in time_list]
            if not time_list_ms:
                continue

            # ── BUG V5 FIX: use first click as RT (matches game metric) ──
            rt_ms       = time_list_ms[0]          # InitialResponseTime equivalent
            total_rt_ms = time_list_ms[-1]          # total completion time (optional)
            n_clicks    = len(time_list_ms)

            avg_inter = (
                np.mean([time_list_ms[i+1] - time_list_ms[i]
                         for i in range(n_clicks - 1)])
                if n_clicks > 1 else 0
            )

            # ── BUG V6 FIX: parse list and use set of target_ names ──
            clicked     = safe_parse_list(row.get("mouse.clicked_name", "[]"))
            unique_hits = len(set(c for c in clicked if str(c).startswith("target_")))
            accuracy    = unique_hits / N_TARGETS_MULTI

            records.append({
                "ParticipantID"    : pid,
                "Group"            : "Multi",
                "Modality"         : "Lab",
                "Trial"            : int(row.get("trials.thisN", len(records))),
                "TargetColour"     : str(row.get("target_col", "")),
                "RT_ms"            : rt_ms,           # first click (consistent with game)
                "TotalRT_ms"       : total_rt_ms,     # last click (optional)
                "FirstClickRT_ms"  : rt_ms,
                "AvgInterTarget_ms": avg_inter,
                "Correct"          : 1 if unique_hits == N_TARGETS_MULTI else 0,
                "NCorrect"         : unique_hits,
                "Accuracy"         : accuracy,
                "FalseAlarms"      : 0,
                "NTargets"         : N_TARGETS_MULTI,
            })
        except Exception:
            continue

    return records


# ─────────────────────────────────────────────────────────────
# PARSER: Single-Target GAME
# ─────────────────────────────────────────────────────────────
def parse_single_game(filepath):
    df  = pd.read_csv(filepath, low_memory=False)
    pid = extract_pid(filepath)

    if "GameMode" in df.columns:
        df = df[df["GameMode"].str.contains("single", case=False, na=False)]

    records = []
    for i, row in df.iterrows():
        try:
            records.append({
                "ParticipantID"  : pid,
                "Group"          : "Single",
                "Modality"       : "Game",
                "Trial"          : int(row.get("Level", i)),
                "Level"          : int(row.get("Level", i)),
                "RT_ms"          : float(row["InitialResponseTime(ms)"]),
                "FirstClickRT_ms": float(row["InitialResponseTime(ms)"]),
                "HitRate_pct"    : float(row.get("HitRate(%)", 100)),
                "SuccessRate_pct": float(row.get("SuccessRate(%)", 100)),
                "Correct"        : 1 if float(row.get("HitRate(%)", 100)) == 100 else 0,
                "FalseAlarms"    : int(row.get("FalseAlarms", 0)),
                "Completed"      : str(row.get("Completed", "true")).lower() == "true",
                "FinalScore"     : int(row.get("FinalScore", 0)),
                "NTargets"       : 1,
            })
        except Exception:
            continue

    return records


# ─────────────────────────────────────────────────────────────
# PARSER: Multiple-Target GAME
# ─────────────────────────────────────────────────────────────
def parse_multi_game(filepath):
    df  = pd.read_csv(filepath, low_memory=False)
    pid = extract_pid(filepath)

    if "GameMode" in df.columns:
        df = df[df["GameMode"].str.contains("multiple", case=False, na=False)]

    records = []
    for i, row in df.iterrows():
        try:
            records.append({
                "ParticipantID"    : pid,
                "Group"            : "Multi",
                "Modality"         : "Game",
                "Trial"            : i,
                "Level"            : int(row.get("Level", i)),
                "RT_ms"            : float(row["InitialResponseTime(ms)"]),
                "FirstClickRT_ms"  : float(row["InitialResponseTime(ms)"]),
                "AvgInterTarget_ms": float(row.get("AvgInterTargetTime(ms)", 0)),
                "HitRate_pct"      : float(row.get("HitRate(%)", 100)),
                "SuccessRate_pct"  : float(row.get("SuccessRate(%)", 100)),
                "Correct"          : 1 if float(row.get("HitRate(%)", 100)) == 100 else 0,
                "FalseAlarms"      : int(row.get("FalseAlarms", 0)),
                "Completed"        : str(row.get("Completed", "true")).lower() == "true",
                "FinalScore"       : int(row.get("FinalScore", 0)),
                "NTargets"         : None,
            })
        except Exception:
            continue

    return records


# ─────────────────────────────────────────────────────────────
# AGGREGATE: Trial-level → Participant-level summary
# ─────────────────────────────────────────────────────────────
def aggregate_participant(trial_records):
    if not trial_records:
        return None

    df       = pd.DataFrame(trial_records)
    pid      = df["ParticipantID"].iloc[0]
    group    = df["Group"].iloc[0]
    modality = df["Modality"].iloc[0]

    summary = {
        "ParticipantID"  : pid,
        "Group"          : group,
        "Modality"       : modality,
        "N_Trials"       : len(df),
        "Mean_RT_ms"     : df["RT_ms"].mean(),
        "SD_RT_ms"       : df["RT_ms"].std(ddof=1),
        "Median_RT_ms"   : df["RT_ms"].median(),
        "Min_RT_ms"      : df["RT_ms"].min(),
        "Max_RT_ms"      : df["RT_ms"].max(),
        "Mean_Accuracy"  : df["Correct"].mean(),
        "Total_FalseAlarms": df["FalseAlarms"].sum(),
    }

    if "FirstClickRT_ms" in df.columns:
        summary["Mean_FirstClickRT_ms"] = df["FirstClickRT_ms"].mean()
        summary["SD_FirstClickRT_ms"]   = df["FirstClickRT_ms"].std(ddof=1)

    if "AvgInterTarget_ms" in df.columns:
        vals = df["AvgInterTarget_ms"].replace(0, np.nan).dropna()
        summary["Mean_AvgInterTarget_ms"] = vals.mean() if len(vals) > 0 else np.nan

    if "Completed" in df.columns:
        summary["N_Completed"]     = df["Completed"].sum()
        summary["Completion_Rate"] = df["Completed"].mean()

    if "Level" in df.columns:
        summary["Max_Level_Reached"] = df["Level"].max()

    if "FinalScore" in df.columns:
        summary["Total_Score"] = df["FinalScore"].sum()

    return summary


# ─────────────────────────────────────────────────────────────
# MAIN: Walk all folders, parse everything, build datasets
# ─────────────────────────────────────────────────────────────
def build_master_dataset(data_root):
    trial_records    = []
    participant_rows = []

    folder_map = {
        os.path.join("single",   "lab")  : ("Single", "Lab",  parse_single_lab),
        os.path.join("single",   "phone"): ("Single", "Game", parse_single_game),
        os.path.join("multiple", "lab")  : ("Multi",  "Lab",  parse_multi_lab),
        os.path.join("multiple", "phone"): ("Multi",  "Game", parse_multi_game),
    }

    print("=" * 65)
    print("  BUILDING MASTER DATASET")
    print(f"  Data root: {os.path.abspath(data_root)}")
    print("=" * 65)

    for rel_path, (group, modality, parser_fn) in folder_map.items():
        folder    = os.path.join(data_root, rel_path)

        if not os.path.isdir(folder):
            print(f"\n⚠  Folder NOT FOUND: {folder}")
            continue

        csv_files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")])
        print(f"\n📂 {rel_path}/ — {len(csv_files)} files found")

        for fname in csv_files:
            fpath = os.path.join(folder, fname)
            pid   = extract_pid(fname)

            try:
                trials  = parser_fn(fpath)
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
# RESHAPE: Long → Wide format
# ─────────────────────────────────────────────────────────────
def make_wide_format(master_long):
    lab  = master_long[master_long["Modality"] == "Lab"].copy()
    game = master_long[master_long["Modality"] == "Game"].copy()

    lab  = lab.add_suffix("_Lab").rename(
        columns={"ParticipantID_Lab": "ParticipantID", "Group_Lab": "Group"})
    game = game.add_suffix("_Game").rename(
        columns={"ParticipantID_Game": "ParticipantID", "Group_Game": "Group"})

    wide = pd.merge(lab, game, on=["ParticipantID", "Group"], how="outer")
    wide["RT_Diff_ms"] = wide["Mean_RT_ms_Lab"] - wide["Mean_RT_ms_Game"]
    wide["Acc_Diff"]   = wide["Mean_Accuracy_Lab"] - wide["Mean_Accuracy_Game"]
    wide["Group_Code"] = wide["Group"].map({"Single": 0, "Multi": 1})

    return wide


# ─────────────────────────────────────────────────────────────
# AUDIT + DESCRIPTIVES + PREVIEW STATS
# ─────────────────────────────────────────────────────────────
def print_data_audit(master_long):
    print("\n" + "=" * 65)
    print("  DATA AUDIT")
    print("=" * 65)
    expected = {"Single": 21, "Multi": 16}
    for group, exp_n in expected.items():
        for mod in ["Lab", "Game"]:
            subset = master_long[(master_long["Group"]==group) &
                                  (master_long["Modality"]==mod)]
            n      = len(subset)
            status = "✓" if n == exp_n else f"⚠ expected {exp_n}"
            print(f"  {group:<6} | {mod:<4} : {n:>2} participants  {status}")
    print(f"\n  Total rows: {len(master_long)}  (expected 74)")


def print_descriptives(master_long):
    print("\n" + "=" * 65)
    print("  DESCRIPTIVE STATISTICS (participant-level means)")
    print("=" * 65)
    print(f"  {'Condition':<20} {'N':>4} {'M_RT (ms)':>12} {'SD_RT':>10}"
          f" {'M_Acc':>8} {'SD_Acc':>8}")
    print("  " + "-" * 63)
    for group in ["Single", "Multi"]:
        for mod in ["Lab", "Game"]:
            sub = master_long[(master_long["Group"]==group) &
                               (master_long["Modality"]==mod)]
            if len(sub) == 0:
                continue
            label = f"{group}-{mod}"
            print(f"  {label:<20} "
                  f"{len(sub):>4} "
                  f"{sub['Mean_RT_ms'].mean():>12.1f} "
                  f"{sub['Mean_RT_ms'].std(ddof=1):>10.1f} "
                  f"{sub['Mean_Accuracy'].mean():>8.3f} "
                  f"{sub['Mean_Accuracy'].std(ddof=1):>8.3f}")


def run_preview_stats(master_long, master_wide):
    print("\n" + "=" * 65)
    print("  PREVIEW STATISTICAL TESTS")
    print("=" * 65)

    print("\n  RQ1 — Concurrent Validity (Pearson r + Spearman rho)")
    for group in ["Single", "Multi"]:
        sub = master_wide[master_wide["Group"]==group].dropna(
            subset=["Mean_RT_ms_Lab", "Mean_RT_ms_Game"])
        if len(sub) < 3:
            print(f"    {group}: insufficient data (N={len(sub)})")
            continue
        x, y = sub["Mean_RT_ms_Lab"].values, sub["Mean_RT_ms_Game"].values
        _, p_nx = stats.shapiro(x)
        _, p_ny = stats.shapiro(y)
        both_norm = (p_nx > 0.05) and (p_ny > 0.05)
        r_p, p_p = stats.pearsonr(x, y)
        r_s, p_s = stats.spearmanr(x, y)
        primary = ("Pearson r", r_p, p_p) if both_norm else ("Spearman rho", r_s, p_s)
        sig = "**" if primary[2] < 0.01 else ("*" if primary[2] < 0.05 else "ns")
        print(f"    {group:<6}: {primary[0]} = {primary[1]:+.3f} ({sig}), "
              f"p = {primary[2]:.4f}, N = {len(sub)}")

    print("\n  RQ2 — Target Load (Independent t-test, Lab RT)")
    lab_s = master_long[(master_long["Group"]=="Single")&
                         (master_long["Modality"]=="Lab")]["Mean_RT_ms"]
    lab_m = master_long[(master_long["Group"]=="Multi")&
                         (master_long["Modality"]=="Lab")]["Mean_RT_ms"]
    if len(lab_s) > 1 and len(lab_m) > 1:
        lev_f, lev_p = stats.levene(lab_s, lab_m)
        eq_var = lev_p > 0.05
        t, p = stats.ttest_ind(lab_s, lab_m, equal_var=eq_var)
        ns, nm = len(lab_s), len(lab_m)
        if eq_var:
            df_t = ns + nm - 2
        else:
            s1, s2 = lab_s.var(ddof=1), lab_m.var(ddof=1)
            df_t = round((s1/ns + s2/nm)**2 /
                         ((s1/ns)**2/(ns-1) + (s2/nm)**2/(nm-1)), 1)
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
        print(f"    Single M={lab_s.mean():.0f} vs Multi M={lab_m.mean():.0f}")
        print(f"    t({df_t}) = {t:.3f}, p = {p:.4f} ({sig})")

    print("\n  RQ3 — Modality Effect (Paired t-test, within groups)")
    for group in ["Single", "Multi"]:
        sub = master_wide[master_wide["Group"]==group].dropna(
            subset=["Mean_RT_ms_Lab", "Mean_RT_ms_Game"])
        if len(sub) < 3:
            continue
        t, p = stats.ttest_rel(sub["Mean_RT_ms_Lab"], sub["Mean_RT_ms_Game"])
        diff = sub["Mean_RT_ms_Lab"].values - sub["Mean_RT_ms_Game"].values
        d    = diff.mean() / diff.std(ddof=1)
        sig  = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
        print(f"    {group:<6}: t({len(sub)-1}) = {t:.3f}, "
              f"p = {p:.4f} ({sig}), d = {d:.3f}, N = {len(sub)}")


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    trial_records, participant_rows = build_master_dataset(DATA_ROOT)

    if not participant_rows:
        print("\n❌ No data parsed. Check DATA_ROOT and folder structure.")
        exit(1)

    master_long = pd.DataFrame(participant_rows)
    master_long = master_long.sort_values(
        ["Group", "ParticipantID", "Modality"]).reset_index(drop=True)

    trial_df = pd.DataFrame(trial_records)
    trial_df = trial_df.sort_values(
        ["Group", "ParticipantID", "Modality", "Trial"]).reset_index(drop=True)

    master_wide = make_wide_format(master_long)
    master_wide = master_wide.sort_values(
        ["Group", "ParticipantID"]).reset_index(drop=True)

    long_path  = os.path.join(OUTPUT_DIR, "master_dataset_long.csv")
    wide_path  = os.path.join(OUTPUT_DIR, "master_dataset_wide.csv")
    trial_path = os.path.join(OUTPUT_DIR, "trial_level_dataset.csv")

    master_long.to_csv(long_path,  index=False)
    master_wide.to_csv(wide_path,  index=False)
    trial_df.to_csv(trial_path,    index=False)

    print_data_audit(master_long)
    print_descriptives(master_long)
    run_preview_stats(master_long, master_wide)

    print("\n" + "=" * 65)
    print("  OUTPUT FILES SAVED")
    print("=" * 65)
    print(f"  📄 {long_path}   — {len(master_long)} rows")
    print(f"  📄 {wide_path}   — {len(master_wide)} rows")
    print(f"  📄 {trial_path}  — {len(trial_df)} rows")
    print("=" * 65)
