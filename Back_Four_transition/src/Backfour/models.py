"""Team-specific classifiers (Table 4), SHAP (Table 5) and their supplementary checks (Tables S4, S5, S9 A)."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .config import FEATURES, N_BOOT, N_DRAWS, NICE, SEED, TEAMS


def make_models(y_train=None, weighted=False):
    """XGBoost, Random Forest and SVM (RBF). weighted=True: class weighting (main analysis);
    XGBoost's scale_pos_weight is the ratio of failures to successes in the training data."""
    if weighted:
        spw = float((y_train == 0).sum()) / float((y_train == 1).sum())
        return {'XGBoost': XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, scale_pos_weight=spw,
                                         random_state=SEED, eval_metric='logloss', verbosity=0),
                'Random Forest': RandomForestClassifier(n_estimators=300, min_samples_split=5,
                                                        class_weight='balanced', random_state=SEED),
                'SVM (RBF)': SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=SEED)}
    return {'XGBoost': XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3,
                                     random_state=SEED, eval_metric='logloss', verbosity=0),
            'Random Forest': RandomForestClassifier(n_estimators=300, min_samples_split=5, random_state=SEED),
            'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=SEED)}


def bootstrap_auc(y, p, n_boot=N_BOOT):
    """Percentile bootstrap 95% CI of the ROC-AUC."""
    r = np.random.default_rng(SEED); y, p = np.asarray(y), np.asarray(p); a = []
    for _ in range(n_boot):
        i = r.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            a.append(roc_auc_score(y[i], p[i]))
    return np.percentile(a, 2.5), np.percentile(a, 97.5)


def fit_team(df, team, seed=SEED, weighted=False, only=None):
    """Fits the models for one team (stratified 80/20 split; scaling inside the pipeline, so it is refitted
    within every cross-validation fold). Returns (rows, fitted pipelines, X_test, split table)."""
    d = df[df['team_lost_possession'] == team].reset_index(drop=True)
    X, y = d[FEATURES], d['label'].astype(int)
    itr, ite = train_test_split(np.arange(len(d)), test_size=0.2, stratify=y, random_state=seed)
    Xtr, Xte, ytr, yte = X.iloc[itr], X.iloc[ite], y.iloc[itr], y.iloc[ite]
    rows, fitted = [], {}
    for name, model in make_models(ytr, weighted).items():
        if only and name != only:
            continue
        pipe = make_pipeline(StandardScaler(), clone(model)).fit(Xtr, ytr)
        fitted[name] = pipe
        p_tr, p_te, pred = pipe.predict_proba(Xtr)[:, 1], pipe.predict_proba(Xte)[:, 1], pipe.predict(Xte)
        cv = cross_val_score(make_pipeline(StandardScaler(), clone(model)), Xtr, ytr,
                             cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring='roc_auc')
        row = {'team': team, 'algorithm': name, 'n train': len(ytr), 'n test': len(yte),
               'train AUC': roc_auc_score(ytr, p_tr), 'test AUC': roc_auc_score(yte, p_te),
               'CV AUC': cv.mean(), 'CV AUC SD': cv.std(ddof=1)}
        if not only:
            lo, hi = bootstrap_auc(yte, p_te)
            row.update({'AUC CI low': lo, 'AUC CI high': hi,
                        'accuracy': accuracy_score(yte, pred), 'precision': precision_score(yte, pred, zero_division=0),
                        'recall': recall_score(yte, pred, zero_division=0), 'F1': f1_score(yte, pred, zero_division=0),
                        'test failure rate': float((yte == 0).mean()),
                        'PR-AUC failure': average_precision_score(1 - yte, 1 - p_te),
                        'failure recall': recall_score(yte, pred, pos_label=0, zero_division=0),
                        'failure precision': precision_score(yte, pred, pos_label=0, zero_division=0),
                        'balanced accuracy': balanced_accuracy_score(yte, pred)})
        rows.append(row)
    split = pd.DataFrame({'team': team, 'match_id': d['match_id'], 'transition_idx': d['transition_idx'],
                          'set': np.where(np.isin(np.arange(len(d)), itr), 'train', 'test')})
    return rows, fitted, Xte, split


def classification_tables(df):
    """Table 4 (class-weighted models on all sequences) and Table 5 (mean |SHAP| on the test set).
    Returns (table4, table5, split indices, fitted pipelines, test sets)."""
    rows, fitted, Xtest, splits = [], {}, {}, []
    for team in TEAMS:
        r, f, xte, sp = fit_team(df, team, weighted=True)
        rows += r; fitted[team] = f; Xtest[team] = xte; splits.append(sp)
    table4 = pd.DataFrame(rows)
    rows = []
    for team in TEAMS:
        for name in ('XGBoost', 'Random Forest'):
            pipe = fitted[team][name]
            for f, imp, sh in zip(FEATURES, pipe[-1].feature_importances_, mean_abs_shap(pipe, Xtest[team])):
                rows.append({'team': team, 'model': name, 'feature': NICE[f], 'importance': imp, 'mean |SHAP|': sh})
    table5 = pd.DataFrame(rows)
    table5['SHAP rank'] = table5.groupby(['team', 'model'])['mean |SHAP|'].rank(ascending=False).astype(int)
    return table4, table5, pd.concat(splits), fitted, Xtest


def mean_abs_shap(pipe, Xte):
    """Mean |SHAP| per indicator: XGBoost's exact TreeSHAP (pred_contribs) or shap.TreeExplainer."""
    import shap
    import xgboost as xgb
    est, Xs = pipe[-1], pipe[:-1].transform(Xte)
    if isinstance(est, XGBClassifier):
        contrib = est.get_booster().predict(xgb.DMatrix(Xs), pred_contribs=True)
        return np.abs(contrib[:, :-1]).mean(axis=0)
    sv = shap.TreeExplainer(est).shap_values(Xs)
    sv = np.asarray(sv[1] if isinstance(sv, list) else sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return np.abs(sv).mean(axis=0)


# ---------------------------------------------------------------- Tables S4 and S5
def balance_50_50(df, seed):
    """Within each team, successes and failures undersampled to the size of the smaller class."""
    parts = []
    for team in TEAMS:
        t = df[df['team_lost_possession'] == team]
        fail, succ = t[t['label'] == 0], t[t['label'] == 1]
        k = min(len(fail), len(succ))
        parts += [fail.sample(k, random_state=seed), succ.sample(k, random_state=seed)]
    return pd.concat(parts).reset_index(drop=True)


def balanced_tables(df, table4):
    """Balanced 50/50 models (seed 42), XGBoost over N_DRAWS balanced draws, and the overfitting check.
    Returns (tableS4, stability, overfitting)."""
    bal = balance_50_50(df, SEED)
    rows = []
    for team in TEAMS:
        r, *_ = fit_team(bal, team)
        rows += r
    tableS4 = pd.DataFrame(rows)
    stab = []
    for s in range(N_DRAWS):
        b = balance_50_50(df, 1000 + s)
        for team in TEAMS:
            r, *_ = fit_team(b, team, seed=1000 + s, only='XGBoost')
            stab.append({'team': team, 'draw': s, 'test AUC': r[0]['test AUC'], 'CV AUC': r[0]['CV AUC']})
    stab = pd.DataFrame(stab)
    both = pd.concat([table4.assign(setting='class-weighted, all (main)'), tableS4.assign(setting='balanced 50/50')])
    both['train - test gap'] = both['train AUC'] - both['test AUC']
    return tableS4, stab, both


def xgboost_auc(df):
    """XGBoost (class-weighted, as Table 4) test and CV ROC-AUC per team; used by the sensitivity analyses."""
    out = []
    for team in TEAMS:
        r, *_ = fit_team(df, team, weighted=True, only='XGBoost')
        out.append({'team': team, 'XGBoost test AUC': r[0]['test AUC'], 'CV AUC': r[0]['CV AUC']})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- Table S9 (A)
def match_grouped_cv(df):
    """Five-fold CV ROC-AUC with folds split by sequence vs by match (no match in both training and
    validation folds); class-weighted models on all of a team's sequences, scaling inside the pipeline."""
    rows = []
    for team in TEAMS:
        d = df[df['team_lost_possession'] == team].reset_index(drop=True)
        X, y, g = d[FEATURES], d['label'].astype(int), d['match_id']
        for name, model in make_models(y, weighted=True).items():
            pipe = make_pipeline(StandardScaler(), clone(model))
            by_seq = cross_val_score(pipe, X, y, scoring='roc_auc',
                                     cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
            by_match = cross_val_score(pipe, X, y, groups=g, scoring='roc_auc',
                                       cv=StratifiedGroupKFold(5, shuffle=True, random_state=SEED))
            rows.append({'team': team, 'algorithm': name,
                         'CV AUC, folds by sequence': by_seq.mean(), 'SD seq': by_seq.std(ddof=1),
                         'CV AUC, folds by match': by_match.mean(), 'SD match': by_match.std(ddof=1)})
    cv_match = pd.DataFrame(rows)
    cv_match['difference (match - seq)'] = cv_match['CV AUC, folds by match'] - cv_match['CV AUC, folds by sequence']
    return cv_match
