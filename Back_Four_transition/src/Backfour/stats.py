"""Statistics: two-way ANOVA (Table 2), mean differences (Table 3), descriptives and outliers (Table S1),
and models accounting for match-level dependence (Table S9 B)."""
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .config import ALPHA_BONF, FEATURES, NICE, TEAMS


# ---------------------------------------------------------------- Table 2
def two_way_anova(df):
    """Team x Outcome ANOVA for each indicator (pingouin, Type II sums of squares)."""
    import pingouin as pg
    rows = []
    for f in FEATURES:
        d = df[['team_lost_possession', 'label', f]].dropna()
        a = pg.anova(dv=f, between=['team_lost_possession', 'label'], data=d, detailed=True)
        pcol = 'p-unc' if 'p-unc' in a.columns else 'p_unc'
        df_err = int(a.loc[a['Source'] == 'Residual', 'DF'].iloc[0])
        for _, x in a[a['Source'] != 'Residual'].iterrows():
            rows.append({'feature': NICE[f],
                         'effect': x['Source'].replace('team_lost_possession', 'Team').replace('label', 'Outcome'),
                         'df': int(x['DF']), 'df_error': df_err, 'F': float(x['F']),
                         'p': float(x[pcol]), 'eta2_p': float(x['np2'])})
    out = pd.DataFrame(rows)
    out['sig .05'] = np.where(out['p'] < 0.05, 'yes', 'no')
    out['sig .01 (Bonf.)'] = np.where(out['p'] < ALPHA_BONF, 'yes', 'no')
    return out


def cell_descriptives(df):
    """Mean, SD and n of each indicator per Team x Outcome cell (raw units)."""
    desc = df.groupby(['team_lost_possession', 'outcome'])[FEATURES].agg(['mean', 'std', 'count'])
    desc.columns = [f"{NICE[f]} {s}" for f, s in desc.columns]
    return desc


# ---------------------------------------------------------------- Table 3
def welch(a, b):
    """Success (a) minus failure (b): difference, 95% CI, t, Welch df, two-sided p."""
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    diff, se = a.mean() - b.mean(), np.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    q, t = stats.t.ppf(0.975, dof), diff / se
    return diff, diff - q * se, diff + q * se, t, dof, 2 * stats.t.sf(abs(t), dof)


def hedges_g(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (1 - 3 / (4 * (na + nb) - 9)) * (a.mean() - b.mean()) / sp


def split_outcome(df, team, f):
    d = df[df['team_lost_possession'] == team]
    return d.loc[d['label'] == 1, f].to_numpy(float), d.loc[d['label'] == 0, f].to_numpy(float)


def mean_differences(df):
    """Success minus failure mean difference with 95% CI (Welch) and Hedges' g, per team and indicator."""
    rows = []
    for team in TEAMS:
        for f in FEATURES:
            a, b = split_outcome(df, team, f)
            diff, lo, hi, *_ = welch(a, b)
            rows.append({'team': team, 'feature': NICE[f], 'n S': len(a), 'n F': len(b),
                         'mean S': a.mean(), 'mean F': b.mean(), 'diff (S - F)': diff,
                         '95% CI low': lo, '95% CI high': hi, 'Hedges g': hedges_g(a, b)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Table S1
def vif(df):
    """Variance inflation factor: 1 / (1 - R^2) of each indicator regressed on the others."""
    out = {}
    for c in df.columns:
        y = df[c].to_numpy(float)
        A = np.column_stack([np.ones(len(df)), df.drop(columns=c).to_numpy(float)])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1 - ((y - A @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out[c] = 1 / (1 - r2)
    return pd.Series(out)


def descriptives_outliers(df):
    """Ranges, z-score and IQR outliers (reported, not removed) and VIF of the five indicators."""
    X = df[FEATURES].rename(columns=NICE)
    s1 = X.describe(percentiles=[.25, .5, .75]).T[['min', '25%', '50%', '75%', 'max', 'mean', 'std']]
    s1.columns = ['Min', 'Q1', 'Median', 'Q3', 'Max', 'Mean', 'SD']
    s1['Range'] = s1['Max'] - s1['Min']
    z = (X - X.mean()) / X.std(ddof=1)
    iqr = s1['Q3'] - s1['Q1']
    lo, hi = s1['Q1'] - 1.5 * iqr, s1['Q3'] + 1.5 * iqr
    s1['z-score outliers (|z|>3)'] = (z.abs() > 3).sum()
    s1['IQR outliers'] = ((X < lo) | (X > hi)).sum()
    s1['IQR lower bound'], s1['IQR upper bound'] = lo, hi
    s1['outlier % (IQR)'] = 100 * s1['IQR outliers'] / len(X)
    s1['VIF'] = vif(X)
    return s1


# ---------------------------------------------------------------- Table S9 (B)
FORMULA = "z ~ C(team, Sum) * C(outcome, Sum)"


def fit_mixed(dm):
    """REML fit with a random intercept for match. Every optimizer is tried and the converged fit with the
    highest REML log-likelihood is kept (some optimizers can stop at a false solution with zero variance)."""
    import statsmodels.formula.api as smf
    fits = []
    for method in (None, 'bfgs', 'lbfgs', 'powell', 'nm', 'cg'):   # None = statsmodels default
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                r = smf.mixedlm(FORMULA, dm, groups=dm['match_id']).fit(reml=True, method=method)
            except Exception:
                continue
        if np.isfinite(r.llf) and np.all(np.isfinite(r.bse_fe)):
            fits.append((bool(r.converged), r.llf, method or 'default', r))
    if not fits:
        raise RuntimeError("no mixed-model fit succeeded")
    fits.sort(key=lambda t: (t[0], t[1]), reverse=True)   # converged first, then highest log-likelihood
    return fits[0][3], fits[0][2]


def _effect_name(term):
    return 'Team * Outcome' if ':' in term else ('Team' if 'team' in term else 'Outcome')


def match_dependence_stats(df, table2):
    """p-values of the ANOVA, of linear mixed models (random intercept for match) and of OLS with
    cluster-robust standard errors (clusters = matches), on z-scaled indicators with sum-coded factors."""
    import statsmodels.formula.api as smf
    dm = df.copy()
    dm['team'] = dm['team_lost_possession']
    dm['outcome'] = dm['label'].map({0: 'Failure', 1: 'Success'})
    clusters = pd.factorize(dm['match_id'])[0]
    rows = []
    for f in FEATURES:
        dm['z'] = (dm[f] - dm[f].mean()) / dm[f].std(ddof=1)
        fit, method = fit_mixed(dm)
        ols = smf.ols(FORMULA, dm).fit(cov_type='cluster', cov_kwds={'groups': clusters})
        var_match = float(fit.cov_re.iloc[0, 0])
        icc = var_match / (var_match + float(fit.scale))
        for term in fit.fe_params.index:
            if term == 'Intercept':
                continue
            rows.append({'feature': NICE[f], 'effect': _effect_name(term),
                         'coef (z units)': float(fit.fe_params[term]),
                         'p mixed model': float(fit.pvalues[term]),
                         'p cluster-robust': float(ols.pvalues[term]),
                         'ICC (match)': icc, 'optimizer': method, 'converged': bool(fit.converged)})
    mixed = pd.DataFrame(rows).merge(table2[['feature', 'effect', 'p']].rename(columns={'p': 'p ANOVA'}),
                                     on=['feature', 'effect'], how='left')
    for col, lab in (('p ANOVA', 'ANOVA'), ('p mixed model', 'mixed'), ('p cluster-robust', 'cluster')):
        mixed[f'sig .01 {lab}'] = np.where(mixed[col] < ALPHA_BONF, 'yes', 'no')
    return mixed
