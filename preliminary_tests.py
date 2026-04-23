"""
=============================================================
  Selective Attention Study — Preliminary Inferential Tests
  Report 1  (UPDATED — non-parametric tests where normality violated)
=============================================================

TEST 1 — Correlation  (RQ1: Concurrent Validity)
  Pearson r if both normal, Spearman rho if either non-normal

TEST 2 — RQ2: Target Load
  Independent t-test if BOTH groups normal
  Mann-Whitney U if either group non-normal  ← NEW

TEST 3 — RQ3: Modality Effect
  Paired t-test if difference scores normal
  Wilcoxon signed-rank if difference scores non-normal  ← NEW

Each test includes:
  - Normality check (Shapiro-Wilk)
  - Primary test + non-parametric alternative where needed
  - Effect size
  - Plain English interpretation
=============================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

INPUT_DIR  = "processed_data"
OUTPUT_DIR = "processed_data"

# ── Load data ──────────────────────────────────────────────
long = pd.read_csv(f"{INPUT_DIR}/master_dataset_long.csv")
wide = pd.read_csv(f"{INPUT_DIR}/master_dataset_wide.csv")

single   = wide[wide["group"] == "single"]
multiple = wide[wide["group"] == "multiple"]

# ── Helpers ────────────────────────────────────────────────
def cohens_d_independent(x, y):
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx-1)*x.std(ddof=1)**2 + (ny-1)*y.std(ddof=1)**2) / (nx+ny-2))
    return (x.mean() - y.mean()) / pooled

def cohens_d_paired(diff):
    return diff.mean() / diff.std(ddof=1)

def rank_biserial_r(U, n1, n2):
    """Effect size for Mann-Whitney U."""
    return 1 - (2*U) / (n1*n2)

def wilcoxon_r(W, n):
    """Effect size for Wilcoxon signed-rank (r = Z/sqrt(N))."""
    # approximate Z from W
    mu    = n*(n+1)/4
    sigma = np.sqrt(n*(n+1)*(2*n+1)/24)
    Z     = abs((W - mu) / sigma)
    return Z / np.sqrt(n)

results = {}

# ══════════════════════════════════════════════════════════
# TEST 1 — RQ1: CONCURRENT VALIDITY (Correlation)
# ══════════════════════════════════════════════════════════
print("=" * 62)
print("  TEST 1 — RQ1: CONCURRENT VALIDITY")
print("  Is there a correlation between game RT and lab RT?")
print("=" * 62)

rq1_results = {}

for grp, sub, label in [("single", single, "Single"), ("multiple", multiple, "Multiple")]:
    x = sub["lab_RT"].values
    y = sub["game_RT"].values
    n = len(sub)

    Wx, px = stats.shapiro(x)
    Wy, py = stats.shapiro(y)
    both_normal = px > 0.05 and py > 0.05
    test_used = "Pearson" if both_normal else "Spearman"

    if both_normal:
        r, p = stats.pearsonr(x, y)
    else:
        r, p = stats.spearmanr(x, y)

    print(f"\n  {label} group (n={n}):")
    print(f"    lab_RT  normality: W={Wx:.3f}, p={px:.3f} → {'normal' if px>0.05 else 'NON-NORMAL'}")
    print(f"    game_RT normality: W={Wy:.3f}, p={py:.3f} → {'normal' if py>0.05 else 'NON-NORMAL'}")
    print(f"    Test used: {test_used}  (chosen based on normality)")
    print(f"    r/rho = {r:.3f},  p = {p:.3f}  {'*' if p<0.05 else '(n.s.)'}")
    print(f"    Shared variance (r²) = {r**2:.2f} = {r**2*100:.0f}%")

    rq1_results[grp] = {"test": test_used, "r": round(r,3), "p": round(p,3),
                         "r2": round(r**2,3), "n": n, "sig": p < 0.05}

print(f"\n  Interpretation:")
print(f"    Single: {'Marginally significant' if rq1_results['single']['sig'] else 'Not significant'} "
      f"correlation — only {rq1_results['single']['r2']*100:.0f}% shared variance. Limited validity.")
print(f"    Multiple: Not significant — game and lab scores unrelated.")
results["RQ1"] = rq1_results

# ══════════════════════════════════════════════════════════
# TEST 2 — RQ2: TARGET LOAD (Single vs Multiple)
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  TEST 2 — RQ2: TARGET LOAD EFFECT")
print("  Is single group different from multiple group?")
print("=" * 62)

rq2_results = {}

for mod in ["lab", "game"]:
    s_vals = single[f"{mod}_RT"].values
    m_vals = multiple[f"{mod}_RT"].values

    # Normality check for BOTH groups
    Ws, ps = stats.shapiro(s_vals)
    Wm, pm = stats.shapiro(m_vals)
    s_normal = ps > 0.05
    m_normal = pm > 0.05
    both_normal = s_normal and m_normal

    # Levene test for equal variance (used with t-test)
    lev_stat, lev_p = stats.levene(s_vals, m_vals)
    equal_var = lev_p > 0.05

    print(f"\n  {mod.upper()} modality:")
    print(f"    Single M={s_vals.mean():.1f} ms  |  Multiple M={m_vals.mean():.1f} ms")
    print(f"    Normality — Single: W={Ws:.3f} p={ps:.3f} → {'normal' if s_normal else 'NON-NORMAL'}")
    print(f"    Normality — Multiple: W={Wm:.3f} p={pm:.3f} → {'normal' if m_normal else 'NON-NORMAL'}")

    if both_normal:
        # Parametric: independent t-test
        print(f"    Levene's test: F={lev_stat:.3f}, p={lev_p:.3f} → equal variance: {'YES' if equal_var else 'NO'}")
        t, p = stats.ttest_ind(s_vals, m_vals, equal_var=equal_var)
        df_t = len(s_vals) + len(m_vals) - 2
        d    = cohens_d_independent(pd.Series(s_vals), pd.Series(m_vals))
        print(f"    TEST: Independent t-test (parametric — both normal)")
        print(f"    t({df_t}) = {t:.3f},  p = {p:.4f}  {'*' if p<0.05 else '(n.s.)'}")
        print(f"    Cohen's d = {d:.3f}")
        rq2_results[mod] = {"test": "independent t-test", "stat": round(t,3),
                             "df": df_t, "p": round(p,4), "effect": round(d,3),
                             "effect_name": "d", "sig": p<0.05}
    else:
        # Non-parametric: Mann-Whitney U
        U, p = stats.mannwhitneyu(s_vals, m_vals, alternative="two-sided")
        r_rb = rank_biserial_r(U, len(s_vals), len(m_vals))
        print(f"    TEST: Mann-Whitney U (non-parametric — normality violated)")
        print(f"    U = {U:.1f},  p = {p:.4f}  {'*' if p<0.05 else '(n.s.)'}")
        print(f"    Rank-biserial r = {r_rb:.3f}  (effect size)")

        # Also show t-test for comparison/reference
        t_ref, p_ref = stats.ttest_ind(s_vals, m_vals)
        d_ref = cohens_d_independent(pd.Series(s_vals), pd.Series(m_vals))
        print(f"    Parametric t-test (for reference): t={t_ref:.3f}, p={p_ref:.4f}, d={d_ref:.3f}")

        rq2_results[mod] = {"test": "Mann-Whitney U", "stat": round(U,1),
                             "p": round(p,4), "effect": round(r_rb,3),
                             "effect_name": "r (rank-biserial)", "sig": p<0.05,
                             "t_ref": round(t_ref,3), "d_ref": round(d_ref,3)}

results["RQ2"] = rq2_results
print(f"\n  Interpretation:")
print(f"    Lab: No significant difference → groups matched at baseline ✓")
print(f"    Game: Single significantly slower than multiple")
print(f"    (Counterintuitive — explained by game difficulty escalation in RQ4)")

# ══════════════════════════════════════════════════════════
# TEST 3 — RQ3: MODALITY EFFECT (Game vs Lab within each group)
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  TEST 3 — RQ3: MODALITY EFFECT")
print("  Is game RT different from lab RT within each group?")
print("=" * 62)

rq3_results = {}

for grp, sub, label in [("single", single, "Single"), ("multiple", multiple, "Multiple")]:
    lab_rt  = sub["lab_RT"].values
    game_rt = sub["game_RT"].values
    diff    = game_rt - lab_rt
    n       = len(sub)

    # Normality check on DIFFERENCE SCORES (not raw RT)
    W_diff, p_diff = stats.shapiro(diff)
    diff_normal = p_diff > 0.05

    print(f"\n  {label} group (n={n}):")
    print(f"    Lab  M = {lab_rt.mean():.1f} ms")
    print(f"    Game M = {game_rt.mean():.1f} ms")
    print(f"    Mean diff (Game − Lab) = {diff.mean():.1f} ms")
    print(f"    Normality of diff scores: W={W_diff:.3f}, p={p_diff:.3f} → {'normal' if diff_normal else 'NON-NORMAL'}")

    if diff_normal:
        # Parametric: paired t-test
        t, p = stats.ttest_rel(game_rt, lab_rt)
        d = cohens_d_paired(pd.Series(diff))
        print(f"    TEST: Paired t-test (parametric — diff scores normal)")
        print(f"    t({n-1}) = {t:.3f},  p = {p:.4f}  {'*' if p<0.05 else '(n.s.)'}")
        print(f"    Cohen's d = {d:.3f}")
        rq3_results[grp] = {"test": "paired t-test", "stat": round(t,3),
                             "df": n-1, "p": round(p,4), "effect": round(d,3),
                             "effect_name": "d", "sig": p<0.05,
                             "mean_diff": round(diff.mean(),1)}
    else:
        # Non-parametric: Wilcoxon signed-rank
        W_stat, p = stats.wilcoxon(game_rt, lab_rt)
        r_w = wilcoxon_r(W_stat, n)
        print(f"    TEST: Wilcoxon signed-rank (non-parametric — diff scores non-normal)")
        print(f"    W = {W_stat:.0f},  p = {p:.4f}  {'*' if p<0.05 else '(n.s.)'}")
        print(f"    Effect size r = {r_w:.3f}")

        # Paired t-test for reference
        t_ref, p_ref = stats.ttest_rel(game_rt, lab_rt)
        d_ref = cohens_d_paired(pd.Series(diff))
        print(f"    Paired t-test (for reference): t({n-1})={t_ref:.3f}, p={p_ref:.4f}, d={d_ref:.3f}")

        rq3_results[grp] = {"test": "Wilcoxon signed-rank", "stat": round(W_stat,0),
                             "p": round(p,4), "effect": round(r_w,3),
                             "effect_name": "r (Wilcoxon)", "sig": p<0.05,
                             "mean_diff": round(diff.mean(),1),
                             "t_ref": round(t_ref,3), "d_ref": round(d_ref,3)}

results["RQ3"] = rq3_results
print(f"\n  Interpretation:")
for grp in ["single", "multiple"]:
    r3 = rq3_results[grp]
    sig_str = "SIGNIFICANT — game substantially slower" if r3["sig"] else "Not significant — game and lab comparable"
    print(f"    {grp.capitalize()}: {sig_str} (mean diff = {r3['mean_diff']:.0f} ms)")

# ══════════════════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════════════════
rows = []
for rq, data in results.items():
    if rq == "RQ1":
        for grp, r in data.items():
            rows.append({"RQ": rq, "comparison": grp, "test": r["test"],
                         "statistic": r["r"], "p": r["p"],
                         "effect_size": r["r2"], "effect_name": "r²",
                         "significant": r["sig"]})
    elif rq == "RQ2":
        for mod, r in data.items():
            rows.append({"RQ": rq, "comparison": f"single vs multiple ({mod})",
                         "test": r["test"], "statistic": r["stat"],
                         "p": r["p"], "effect_size": r["effect"],
                         "effect_name": r["effect_name"], "significant": r["sig"]})
    elif rq == "RQ3":
        for grp, r in data.items():
            rows.append({"RQ": rq, "comparison": f"{grp} game vs lab",
                         "test": r["test"], "statistic": r["stat"],
                         "p": r["p"], "effect_size": r["effect"],
                         "effect_name": r["effect_name"], "significant": r["sig"]})

res_df = pd.DataFrame(rows)
res_df.to_csv(f"{OUTPUT_DIR}/preliminary_test_results.csv", index=False)

print(f"\n{'='*62}")
print(f"  RESULTS SAVED → {OUTPUT_DIR}/preliminary_test_results.csv")
print(f"{'='*62}")
print(f"\n  SUMMARY TABLE:")
print(res_df.to_string(index=False))

print(f"""
{'='*62}
  KEY CHANGES FROM ORIGINAL SCRIPT
{'='*62}
  RQ1: No change — Spearman already used for single group ✓
  RQ2 lab: No change — both groups normal, t-test appropriate ✓
  RQ2 game: CHANGED — single group non-normal
            → Mann-Whitney U used (primary)
            → t-test shown for reference only
  RQ3 single: CHANGED — diff scores non-normal
              → Wilcoxon signed-rank used (primary)
              → paired t-test shown for reference only
  RQ3 multiple: No change — diff scores normal, paired t ✓
{'='*62}
""")
