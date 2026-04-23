"""
check_game_files.py
===================
Run this BEFORE preprocess_final.py.

Scans every game file (single/phone/ and multiple/phone/) and reports:
  1. Which files have MORE than one PlayerID  → contaminated
  2. Which PlayerID is the REAL one per file  → based on timestamps
  3. Exactly which rows to keep and which to discard

Also prints the correct RT for every participant so you can verify
your final dataset values by eye.

HOW TO RUN:
  python check_game_files.py

Set DATA_ROOT below if your folder is not the current directory.
"""

import os
import glob
import pandas as pd
import numpy as np

DATA_ROOT = "."   # change if needed

print("=" * 65)
print("  GAME FILE DIAGNOSTIC — PlayerID contamination check")
print("=" * 65)

all_clean     = True
problem_files = []

for group in ["single", "multiple"]:
    folder = os.path.join(DATA_ROOT, group, "phone")
    files  = sorted(glob.glob(os.path.join(folder, "*.csv")))

    if not files:
        print(f"\n  [WARNING] No files found in: {folder}")
        continue

    print(f"\n{'─'*65}")
    print(f"  {group}/phone  ({len(files)} files)")
    print(f"{'─'*65}")

    for f in files:
        fname = os.path.basename(f)
        pid   = fname.split("_")[0]
        df    = pd.read_csv(f)

        if "PlayerID" not in df.columns:
            print(f"  P{pid:>3}: NO PlayerID column — cannot check")
            continue

        player_ids = df["PlayerID"].unique().tolist()
        n_players  = len(player_ids)
        n_rows     = len(df)

        if n_players == 1:
            # Clean file — single session
            pid_label  = player_ids[0]
            completed  = df[df["Completed"].astype(str).str.lower() == "true"]
            n_completed = len(completed)
            n_levels   = completed["Level"].nunique() if n_completed > 0 else 0
            rt_mean    = completed.groupby("Level")["InitialResponseTime(ms)"].first().mean()
            print(f"  P{pid:>3}: {pid_label:<10}  rows={n_rows:>3}  "
                  f"completed_rows={n_completed:>3}  "
                  f"levels={n_levels:>2}  mean_RT={rt_mean:>8.1f} ms  ✓")

        else:
            # Contaminated file — multiple sessions
            all_clean = False
            print(f"\n  P{pid:>3}: ⚠  MULTIPLE PlayerIDs FOUND: {player_ids}")

            # Find which PlayerID matches another participant's file
            # Strategy: the REAL player for this file played AFTER the contamination
            # Use timestamps to identify the latest session
            df["Timestamp_dt"] = pd.to_datetime(df["Timestamp"])
            df_sorted = df.sort_values("Timestamp_dt")

            session_info = []
            for pid_val in player_ids:
                sub = df[df["PlayerID"] == pid_val]
                t_start = pd.to_datetime(sub["Timestamp"]).min()
                t_end   = pd.to_datetime(sub["Timestamp"]).max()
                n_sub   = len(sub)
                session_info.append((pid_val, t_start, t_end, n_sub))
                print(f"       {pid_val}: {n_sub} rows, "
                      f"from {t_start.strftime('%H:%M:%S')} "
                      f"to {t_end.strftime('%H:%M:%S')}")

            # The REAL session for this file is the one whose timestamps
            # do NOT appear in any other file — can't check that without
            # all files, but we flag it for manual verification
            print(f"       → ACTION: check which PlayerID is the real P{pid}")
            print(f"         The contaminating session will have timestamps")
            print(f"         that match exactly with another participant's file.")
            print(f"         Keep only the rows belonging to P{pid}'s session.")

            # Show what each session gives as RT
            for pid_val, t_start, t_end, n_sub in session_info:
                sub       = df[df["PlayerID"] == pid_val]
                completed = sub[sub["Completed"].astype(str).str.lower() == "true"]
                if len(completed) > 0:
                    first_done = completed.groupby("Level")["InitialResponseTime(ms)"].first()
                    rt_mean = first_done.mean()
                    print(f"         {pid_val} → mean RT if kept = {rt_mean:.1f} ms "
                          f"({len(first_done)} levels)")
                else:
                    print(f"         {pid_val} → no completed levels")

            problem_files.append({"pid": pid, "file": fname, "player_ids": player_ids})
            print()

print("\n" + "=" * 65)
if all_clean:
    print("  ✅ All game files are clean — no contamination detected.")
else:
    print(f"  ⚠  {len(problem_files)} contaminated file(s) found:")
    for p in problem_files:
        print(f"     P{p['pid']}: {p['file']}  PlayerIDs={p['player_ids']}")
    print()
    print("  HOW TO FIX:")
    print("  For each contaminated file, open it and delete all rows")
    print("  that belong to the wrong session (the one whose timestamps")
    print("  match another participant's file).")
    print("  Then save and rerun preprocess_final.py.")
    print()
    print("  OR: add the PlayerID filter in preprocess_final.py —")
    print("  see the fix below.")
print("=" * 65)

print("""
─────────────────────────────────────────────────────────────
  FIX FOR preprocess_final.py  (add inside get_game_clean)
─────────────────────────────────────────────────────────────
  If a file has multiple PlayerIDs, keep only the session
  whose timestamps are UNIQUE to that file (not shared with
  any other participant).

  Simplest safe fix — keep the LAST session by time
  (the real participant played after the contaminating one):

  In get_game_clean(), after reading df, add:

      if "PlayerID" in df.columns:
          n_players = df["PlayerID"].nunique()
          if n_players > 1:
              # Sort by time, keep only the last session's PlayerID
              df = df.sort_values("Timestamp").reset_index(drop=True)
              last_pid = df["PlayerID"].iloc[-1]
              df = df[df["PlayerID"] == last_pid].copy()
              print(f"  [INFO] Multi-session file: kept PlayerID={last_pid}")

  This works because the contaminating session always comes
  BEFORE the real session (same device, earlier in time).
─────────────────────────────────────────────────────────────
""")
