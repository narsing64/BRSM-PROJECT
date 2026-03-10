"""
=============================================================
  Selective Attention Study — Preliminary Inferential Tests
  Report 1
=============================================================

TEST 1 — Pearson r + Spearman rho  (RQ1: Concurrent Validity)
  Is Lab RT correlated with Game RT per group?

TEST 2 — Independent Samples t-test  (RQ2: Target Load)
  Is RT significantly different between Single and Multiple groups?

TEST 3 — Paired Samples t-test  (RQ3: Modality Effect)
  Is Game RT significantly different from Lab RT within each group?

Each test includes:
  - Assumption checks
  - Test statistic + p-value
  - Effect size (Cohen's d / r)
  - Interpretation for report
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
wide = long.pivot_table(
    index=["participant", "group"],
    columns="modality",
    values=["RT_ms", "accuracy"]
).reset_index()
wide.columns = ["participant", "group", "game_RT", "lab_RT", "game_acc", "lab_acc"]

# ── Helper: Cohen's d ──────────────────────────────────────
def cohens_d_independent(x, y):
    """Pooled Cohen's d for independent samples."""
    nx, ny = len(x), len(y)
    pooled_sd = np.sqrt(
        ((nx - 1) * x.std(ddof=1)**2 + (ny - 1) * y.std(ddof=1)**2)
        / (nx + ny - 2)
    )
    return (x.mean() - y.mean()) / pooled_sd

def cohens_d_paired(x, y):
    """Cohen's d for paired samples (mean diff / SD of diff)."""
    diff = x - y
    return diff.mean() / diff.std(ddof=1)

def effect_label(d):
    """Interpret Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:   return "negligible"
    elif d < 0.5: return "small"
    elif d < 0.8: return "medium"
    else:         return "large"

def r_to_label(r):
    """Interpret Pearson/Spearman r magnitude."""
    r = abs(r)
    if r < 0.10:  return "negligible"
    elif r < 0.30: return "small"
    elif r < 0.50: return "moderate"
    else:          return "strong"

# Results accumulator for CSV
results_rows = []

print("=" * 62)
print("  PRELIMINARY INFERENTIAL TESTS — REPORT 1")
print("=" * 62)


# ═════════════════════════════════════════════════════════════
# TEST 1 — CORRELATION  (RQ1: Concurrent Validity)
# Is there a significant positive correlation between Game RT
# and Lab RT within each group?
# ═════════════════════════════════════════════════════════════
print("\n── TEST 1: Correlation — Lab RT vs Game RT (RQ1) ───────")
print("  H0: No correlation between Lab RT and Game RT")
print("  H1: Significant positive correlation exists")
print()

for grp in ["single", "multiple"]:
    sub = wide[wide["group"] == grp]
    x   = sub["lab_RT"].values
    y   = sub["game_RT"].values
    n   = len(sub)

    # ── Assumption: normality of both variables ──
    w_x, p_x = stats.shapiro(x)
    w_y, p_y = stats.shapiro(y)
    both_normal = (p_x > 0.05) and (p_y > 0.05)

    print(f"  [{grp.upper()} GROUP  n={n}]")
    print(f"  Normality check:")
    print(f"    Lab RT:  W={w_x:.3f}, p={p_x:.4f} "
          f"→ {'normal ✓' if p_x>0.05 else 'non-normal ⚠'}")
    print(f"    Game RT: W={w_y:.3f}, p={p_y:.4f} "
          f"→ {'normal ✓' if p_y>0.05 else 'non-normal ⚠'}")
    print()

    # ── Pearson r ──
    r_p, p_p = stats.pearsonr(x, y)
    # ── Spearman rho (non-parametric alternative) ──
    r_s, p_s = stats.spearmanr(x, y)

    # Report both when normality is violated; otherwise Pearson is primary
    primary = "Spearman" if not both_normal else "Pearson"

    print(f"  Pearson  r   = {r_p:+.3f},  p = {p_p:.4f}  "
          f"({'significant *' if p_p<0.05 else 'not significant'})"
          f"  [{r_to_label(r_p)} effect]")
    print(f"  Spearman rho = {r_s:+.3f},  p = {p_s:.4f}  "
          f"({'significant *' if p_s<0.05 else 'not significant'})"
          f"  [{r_to_label(r_s)} effect]")
    print(f"  → Primary statistic: {primary} (based on normality)")
    print()

    # ── Interpretation ──
    r_use = r_s if not both_normal else r_p
    p_use = p_s if not both_normal else p_p
    if p_use < 0.05:
        interp = (f"There was a statistically significant positive correlation "
                  f"between Lab RT and Game RT for the {grp} group "
                  f"({primary} {'rho' if primary=='Spearman' else 'r'} = {r_use:.3f}, "
                  f"p = {p_use:.3f}), indicating concurrent validity.")
    else:
        interp = (f"There was no statistically significant correlation "
                  f"between Lab RT and Game RT for the {grp} group "
                  f"({primary} {'rho' if primary=='Spearman' else 'r'} = {r_use:.3f}, "
                  f"p = {p_use:.3f}), suggesting limited concurrent validity.")
    print(f"  Interpretation: {interp}")
    print()

    results_rows.append({
        "test"       : "Correlation",
        "comparison" : f"{grp}: Lab RT vs Game RT",
        "n"          : n,
        "statistic"  : f"Pearson r={r_p:.3f}; Spearman rho={r_s:.3f}",
        "p_value"    : round(p_p, 4),
        "effect_size": f"r={r_p:.3f} ({r_to_label(r_p)})",
        "significant": "Yes" if p_p < 0.05 else "No",
        "RQ"         : "RQ1",
    })


# ═════════════════════════════════════════════════════════════
# TEST 2 — INDEPENDENT SAMPLES t-TEST  (RQ2: Target Load)
# Is RT significantly worse in Multiple Target vs Single Target?
# ═════════════════════════════════════════════════════════════
print("\n── TEST 2: Independent t-test — Single vs Multiple (RQ2) ─")
print("  H0: No difference in RT between Single and Multiple groups")
print("  H1: Multiple group RT ≠ Single group RT")
print()

for mod in ["game", "lab"]:
    s_rt = long[(long["group"] == "single")   & (long["modality"] == mod)]["RT_ms"]
    m_rt = long[(long["group"] == "multiple") & (long["modality"] == mod)]["RT_ms"]
    ns, nm = len(s_rt), len(m_rt)

    print(f"  [{mod.upper()} MODALITY]")

    # ── Assumption: equal variances (Levene's test) ──
    lev_f, lev_p = stats.levene(s_rt, m_rt)
    equal_var = lev_p > 0.05
    print(f"  Levene's test: F={lev_f:.3f}, p={lev_p:.4f} "
          f"→ {'equal variances ✓' if equal_var else 'unequal variances → Welch t-test'}")

    # ── t-test (Welch if unequal variance) ──
    t, p = stats.ttest_ind(s_rt, m_rt, equal_var=equal_var)
    df_t = (ns + nm - 2) if equal_var else None   # approx for Welch
    d    = cohens_d_independent(s_rt, m_rt)

    print(f"  Single:   M={s_rt.mean():.1f} ms, SD={s_rt.std(ddof=1):.1f}, n={ns}")
    print(f"  Multiple: M={m_rt.mean():.1f} ms, SD={m_rt.std(ddof=1):.1f}, n={nm}")
    print(f"  t = {t:.3f},  p = {p:.4f}  "
          f"({'significant *' if p<0.05 else 'not significant'})")
    print(f"  Cohen's d = {d:.3f}  [{effect_label(d)} effect]")
    print()

    # ── Interpretation ──
    if p < 0.05:
        faster = "single" if s_rt.mean() < m_rt.mean() else "multiple"
        interp = (f"RT was significantly {'lower' if s_rt.mean()<m_rt.mean() else 'higher'} "
                  f"in the single group compared to the multiple group "
                  f"in the {mod} condition (t = {t:.3f}, p = {p:.4f}, d = {d:.3f}).")
    else:
        interp = (f"No significant difference in RT between groups "
                  f"in the {mod} condition (t = {t:.3f}, p = {p:.4f}, d = {d:.3f}).")
    print(f"  Interpretation: {interp}")
    print()

    results_rows.append({
        "test"       : "Independent t-test",
        "comparison" : f"Single vs Multiple — {mod} RT",
        "n"          : f"n_single={ns}, n_multi={nm}",
        "statistic"  : f"t={t:.3f}",
        "p_value"    : round(p, 4),
        "effect_size": f"d={d:.3f} ({effect_label(d)})",
        "significant": "Yes" if p < 0.05 else "No",
        "RQ"         : "RQ2",
    })


# ═════════════════════════════════════════════════════════════
# TEST 3 — PAIRED SAMPLES t-TEST  (RQ3: Modality Effect)
# Does the gamified interface significantly alter RT compared
# to the standard lab task?
# ═════════════════════════════════════════════════════════════
print("\n── TEST 3: Paired t-test — Game vs Lab (RQ3) ───────────")
print("  H0: No difference in RT between Game and Lab modalities")
print("  H1: Game RT ≠ Lab RT (within each group)")
print()

for grp in ["single", "multiple"]:
    sub     = wide[wide["group"] == grp].copy()
    game_rt = sub["game_RT"].values
    lab_rt  = sub["lab_RT"].values
    n       = len(sub)
    diff    = game_rt - lab_rt

    print(f"  [{grp.upper()} GROUP  n={n}]")

    # ── Assumption: normality of difference scores ──
    w_d, p_d = stats.shapiro(diff)
    print(f"  Normality of differences: W={w_d:.3f}, p={p_d:.4f} "
          f"→ {'normal ✓' if p_d>0.05 else 'non-normal ⚠'}")

    # ── Paired t-test ──
    t, p = stats.ttest_rel(game_rt, lab_rt)
    d    = cohens_d_paired(game_rt, lab_rt)

    print(f"  Lab RT:  M={lab_rt.mean():.1f} ms, SD={lab_rt.std(ddof=1):.1f}")
    print(f"  Game RT: M={game_rt.mean():.1f} ms, SD={game_rt.std(ddof=1):.1f}")
    print(f"  Mean difference (Game − Lab): {diff.mean():.1f} ms "
          f"(SD={diff.std(ddof=1):.1f})")
    print(f"  t({n-1}) = {t:.3f},  p = {p:.4f}  "
          f"({'significant *' if p<0.05 else 'not significant'})")
    print(f"  Cohen's d = {d:.3f}  [{effect_label(d)} effect]")
    print()

    # ── Interpretation ──
    if p < 0.05:
        direction = "higher" if diff.mean() > 0 else "lower"
        interp = (f"Game RT was significantly {direction} than Lab RT "
                  f"for the {grp} group "
                  f"(t({n-1}) = {t:.3f}, p = {p:.4f}, d = {d:.3f}), "
                  f"suggesting the gamified interface "
                  f"{'slowed' if direction=='higher' else 'sped up'} responses.")
    else:
        interp = (f"No significant difference between Game and Lab RT "
                  f"for the {grp} group "
                  f"(t({n-1}) = {t:.3f}, p = {p:.4f}, d = {d:.3f}).")
    print(f"  Interpretation: {interp}")
    print()

    results_rows.append({
        "test"       : "Paired t-test",
        "comparison" : f"{grp}: Game RT vs Lab RT",
        "n"          : n,
        "statistic"  : f"t({n-1})={t:.3f}",
        "p_value"    : round(p, 4),
        "effect_size": f"d={d:.3f} ({effect_label(d)})",
        "significant": "Yes" if p < 0.05 else "No",
        "RQ"         : "RQ3",
    })


# ═════════════════════════════════════════════════════════════
# SAVE RESULTS TABLE
# ═════════════════════════════════════════════════════════════
results_df = pd.DataFrame(results_rows)
out_path   = f"{OUTPUT_DIR}/preliminary_test_results.csv"
results_df.to_csv(out_path, index=False)

print("\n" + "=" * 62)
print("  RESULTS SUMMARY TABLE")
print("=" * 62)
print(results_df[["RQ","test","comparison","statistic",
                   "p_value","effect_size","significant"]]
      .to_string(index=False))
print(f"\n✅  Results saved → {out_path}")


# ═════════════════════════════════════════════════════════════
# REPORT-READY PARAGRAPH SUMMARY
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  REPORT-READY SUMMARY (paste into Results section)")
print("=" * 62)

single  = wide[wide["group"] == "single"]
multi   = wide[wide["group"] == "multiple"]
s_game  = long[(long["group"]=="single")  &(long["modality"]=="game")]["RT_ms"]
m_game  = long[(long["group"]=="multiple")&(long["modality"]=="game")]["RT_ms"]

r_s, p_s_corr = stats.pearsonr(single["lab_RT"], single["game_RT"])
rho_s, p_s_sp = stats.spearmanr(single["lab_RT"], single["game_RT"])
r_m, p_m_corr = stats.pearsonr(multi["lab_RT"],  multi["game_RT"])

t_rq2, p_rq2  = stats.ttest_ind(s_game, m_game, equal_var=True)
d_rq2         = cohens_d_independent(s_game, m_game)

t_s, p_s_pair = stats.ttest_rel(single["game_RT"], single["lab_RT"])
d_s           = cohens_d_paired(single["game_RT"].values, single["lab_RT"].values)
t_m, p_m_pair = stats.ttest_rel(multi["game_RT"],  multi["lab_RT"])
d_m           = cohens_d_paired(multi["game_RT"].values,  multi["lab_RT"].values)

print(f"""
RQ1 — Concurrent Validity:
  Pearson correlations between Lab RT and Game RT were weak and
  non-significant for both the single-target group (r = {r_s:.3f},
  p = {p_s_corr:.3f}) and the multiple-target group (r = {r_m:.3f},
  p = {p_m_corr:.3f}). Given the non-normality of single-target lab
  RT (Shapiro-Wilk W = 0.869, p = .009), Spearman's rho was also
  computed for the single group (rho = {rho_s:.3f}, p = {p_s_sp:.3f}),
  which similarly showed no significant correlation. These results
  suggest limited concurrent validity between the game and lab task.

RQ2 — Target Load Effect:
  An independent samples t-test revealed a significant difference
  in game RT between the single-target (M = {s_game.mean():.0f} ms,
  SD = {s_game.std(ddof=1):.0f}) and multiple-target groups
  (M = {m_game.mean():.0f} ms, SD = {m_game.std(ddof=1):.0f}),
  t({len(s_game)+len(m_game)-2}) = {t_rq2:.3f}, p = {p_rq2:.4f},
  d = {d_rq2:.3f} (large effect). Single-target participants were
  significantly slower in the game, suggesting greater difficulty
  locating a single target among distractors compared to multiple
  salient targets.

RQ3 — Modality Effect:
  Paired t-tests compared Game RT to Lab RT within each group.
  For the single-target group, Game RT was significantly higher
  than Lab RT (t({len(single)-1}) = {t_s:.3f}, p < .001,
  d = {d_s:.3f}, large effect), indicating the game interface
  substantially slowed responses. For the multiple-target group,
  the difference was not significant (t({len(multi)-1}) = {t_m:.3f},
  p = {p_m_pair:.3f}, d = {d_m:.3f}, small effect), suggesting
  the game and lab tasks produced comparable RT in this group.
""")