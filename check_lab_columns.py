"""
check_lab_columns.py
--------------------
Run this from your data_brsm folder.
It scans every lab (visual_search) CSV and reports:
  - whether mouse.time exists
  - whether click_times exists
  - which group (single/multiple) based on folder
  - what the actual values look like for trial 1
"""

import os, glob, ast
import pandas as pd
import numpy as np

# ── adjust this path if needed ──────────────────────────────
DATA_ROOT = "."   # run from inside data_brsm folder
# ────────────────────────────────────────────────────────────

single_dir   = os.path.join(DATA_ROOT, "single", "lab")
multiple_dir = os.path.join(DATA_ROOT, "multiple", "lab")

results = []

for group, folder in [("single", single_dir), ("multiple", multiple_dir)]:
    files = sorted(glob.glob(os.path.join(folder, "*visual_search*.csv")))
    for fpath in files:
        pid = os.path.basename(fpath).split("_")[0]
        try:
            df = pd.read_csv(fpath)
            trials = df[df['target_col'].notna() & (df['target_col'].astype(str).str.strip() != '')]

            has_mouse_time   = 'mouse.time'   in df.columns
            has_click_times  = 'click_times'  in df.columns

            # get first trial value for whichever columns exist
            mt_val  = None
            ct_val  = None
            mt_rt   = None
            ct_rt   = None

            if has_mouse_time and len(trials) > 0:
                raw = str(trials.iloc[0]['mouse.time'])
                try:
                    parsed = ast.literal_eval(raw)
                    mt_val = parsed[0]
                    mt_rt  = round(mt_val * 1000, 1)
                except:
                    mt_val = raw

            if has_click_times and len(trials) > 0:
                raw = str(trials.iloc[0]['click_times'])
                try:
                    parsed = ast.literal_eval(raw)
                    ct_val = parsed[0]
                    ct_rt  = round(ct_val * 1000, 1)
                except:
                    ct_val = raw

            results.append({
                "pid"             : pid,
                "group"           : group,
                "n_trials"        : len(trials),
                "has_mouse.time"  : "YES" if has_mouse_time  else "NO",
                "has_click_times" : "YES" if has_click_times else "NO",
                "mouse.time[0]*1000 (ms)" : mt_rt,
                "click_times[0]*1000 (ms)": ct_rt,
                "diff (ct-mt) ms" : round(ct_rt - mt_rt, 1) if (ct_rt and mt_rt) else "N/A"
            })

        except Exception as e:
            results.append({"pid": pid, "group": group, "error": str(e)})

res_df = pd.DataFrame(results)

print("\n" + "="*90)
print("  LAB FILE COLUMN AUDIT — all participants")
print("="*90)
print(res_df.to_string(index=False))

print("\n\n=== SUMMARY ===")
single_res   = res_df[res_df['group']=='single']
multiple_res = res_df[res_df['group']=='multiple']

print(f"\nSingle target  ({len(single_res)} files):")
print(f"  mouse.time present  : {(single_res['has_mouse.time']=='YES').sum()} / {len(single_res)}")
print(f"  click_times present : {(single_res['has_click_times']=='YES').sum()} / {len(single_res)}")

print(f"\nMultiple target ({len(multiple_res)} files):")
print(f"  mouse.time present  : {(multiple_res['has_mouse.time']=='YES').sum()} / {len(multiple_res)}")
print(f"  click_times present : {(multiple_res['has_click_times']=='YES').sum()} / {len(multiple_res)}")

print("\n\n=== CONCLUSION ===")
single_has_ct   = (single_res['has_click_times']=='YES').all()
multiple_has_ct = (multiple_res['has_click_times']=='YES').all()
single_has_mt   = (single_res['has_mouse.time']=='YES').all()
multiple_has_mt = (multiple_res['has_mouse.time']=='YES').all()

if not single_has_ct and multiple_has_ct:
    print("✓ Single target files: NO click_times — only mouse.time available")
    print("✓ Multiple target files: BOTH click_times and mouse.time present")
    print()
    print("→ USE mouse.time[0] × 1000 for BOTH groups (only common column)")
    print("  This keeps the reference point consistent for fair comparison")
elif single_has_ct and multiple_has_ct:
    print("✓ Both groups have click_times")
    print("→ USE click_times[0] × 1000 for both — it is the more accurate RT")
elif single_has_mt and multiple_has_mt:
    print("✓ Both groups have mouse.time")
    print("→ USE mouse.time[0] × 1000 for both")
else:
    print("⚠ Mixed situation — check individual files above")
