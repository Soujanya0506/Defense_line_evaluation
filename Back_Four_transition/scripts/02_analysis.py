"""Step 2: all tables of the paper from the sequence features.

    python scripts/02_analysis.py --out-dir outputs

Reads outputs/features.csv and outputs/features_preloss.csv (step 1) and writes one CSV per table:
Table 2 (two-way ANOVA), Table 3 (mean differences), Table 4 (classification), Table 5 (SHAP),
Supplementary Tables S1 to S9. About 5 minutes (most of it the 20 balanced draws).
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from backfour import models, stats  # noqa: E402
from backfour.pipeline import analysis_sample  # noqa: E402

pd.set_option('display.width', 160)


def save(df, out_dir, name, index=False):
    df.to_csv(os.path.join(out_dir, name), index=index)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', default='outputs')
    args = ap.parse_args()
    out = args.out_dir

    sample = analysis_sample(pd.read_csv(os.path.join(out, 'features.csv')))
    print(f"analysis sample: {len(sample):,} sequences")
    print(sample.groupby(['team_lost_possession', 'outcome']).size().unstack().to_string(), "\n")

    # ---- Table 2 and cell means ----
    save(stats.cell_descriptives(sample), out, 'cell_descriptives.csv', index=True)
    table2 = stats.two_way_anova(sample)
    save(table2, out, 'table2_anova.csv')
    print("Table 2 (two-way ANOVA):\n", table2.round(4).to_string(), "\n")

    # ---- Table 3 ----
    table3 = stats.mean_differences(sample)
    save(table3, out, 'table3_mean_differences.csv')
    print("Table 3 (success - failure, 95% CI):\n", table3.round(3).to_string(), "\n")

    # ---- Tables 4 and 5 ----
    table4, table5, splits, _, _ = models.classification_tables(sample)
    save(table4, out, 'table4_class_weighted.csv')
    save(splits, out, 'table4_split_indices.csv')
    save(table5, out, 'table5_shap.csv')
    print("Table 4 (class-weighted models):\n",
          table4[['team', 'algorithm', 'test AUC', 'AUC CI low', 'AUC CI high', 'CV AUC', 'CV AUC SD',
                  'balanced accuracy', 'failure recall', 'failure precision', 'PR-AUC failure']].round(3).to_string(), "\n")

    # ---- Table S1 ----
    save(stats.descriptives_outliers(sample), out, 'tableS1_descriptives_outliers.csv', index=True)

    # ---- Tables S4 and S5: balanced 50/50, stability, overfitting ----
    tableS4, stab, both = models.balanced_tables(sample, table4)
    save(tableS4, out, 'tableS4_balanced.csv')
    save(stab, out, 'tableS4_stability.csv')
    save(both, out, 'tableS4_overfitting.csv')

    # ---- Table S6: ball reached the own half by the 5th event ----
    early = sample[sample['entry_event_own_half'] <= 5].copy()
    by_event = sample.groupby('entry_event_own_half')['label'].agg(n='size', failure=lambda x: 1 - x.mean())
    save(by_event, out, 'robust_censoring_by_event.csv', index=True)
    cmp_ = pd.concat([stats.two_way_anova(sample).assign(sample=f'main (n={len(sample)})'),
                      stats.two_way_anova(early).assign(sample=f'by event 5 (n={len(early)})')])
    save(cmp_, out, 'robust_censoring.csv')

    # ---- Table S7: indicators measured only before the possession loss ----
    pre_path = os.path.join(out, 'features_preloss.csv')
    if os.path.exists(pre_path):
        key = ['match_id', 'transition_idx']
        pre = pd.read_csv(pre_path).merge(sample[key], on=key)
        pre['outcome'] = pre['label'].map({0: 'Failure', 1: 'Success'})
        if len(pre) != len(sample):
            print(f"WARNING: {len(pre)} of {len(sample)} sequences found in features_preloss.csv")
        cmp2 = pd.concat([stats.two_way_anova(sample).assign(window='outcome window (main)'),
                          stats.two_way_anova(pre).assign(window='before the loss')])
        save(cmp2, out, 'robust_preloss.csv')
        save(models.xgboost_auc(pre), out, 'robust_preloss_auc.csv')
    else:
        print("features_preloss.csv not found: Table S7 skipped (run step 1 without --skip-preloss)")

    # ---- Table S8: alternative failure definition (controlled entry) ----
    sB = sample[sample['valid_B'].astype(bool)].copy()
    sB['label'] = sB['label_B']
    sB['outcome'] = sB['label'].map({0: 'Failure', 1: 'Success'})
    cmpB = pd.concat([table2.assign(definition='A (main)'),
                      stats.two_way_anova(sB).assign(definition='B (controlled entry)')])
    save(cmpB, out, 'robust_outcome_B.csv')
    save(models.xgboost_auc(sB), out, 'robust_outcome_B_auc.csv')

    # ---- Table S9: match-level dependence ----
    cv_match = models.match_grouped_cv(sample)
    save(cv_match, out, 'robust_match_grouped_cv.csv')
    mixed = stats.match_dependence_stats(sample, table2)
    save(mixed, out, 'robust_mixed_model.csv')
    print("Table S9 (B): p-values\n",
          mixed[['feature', 'effect', 'p ANOVA', 'p mixed model', 'p cluster-robust', 'ICC (match)',
                 'converged']].round(4).to_string())
    print("\nall mixed models converged:", bool(mixed['converged'].all()))
    print(f"tables written to {os.path.abspath(out)}")


if __name__ == '__main__':
    main()
