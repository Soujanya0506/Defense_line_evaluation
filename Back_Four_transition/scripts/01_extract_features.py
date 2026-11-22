"""Step 1: one row per defensive sequence, with its label and indicators.

    python scripts/01_extract_features.py --data-dir /path/to/matches --out-dir outputs

Writes outputs/features.csv (main analysis) and outputs/features_preloss.csv (indicators measured only
in the 12 frames before the possession loss; robustness analysis 2), plus a log of kept and excluded
turnovers. About 10 minutes per run for 73 matches.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from backfour.pipeline import analysis_sample, extract_features  # noqa: E402
from backfour.config import TEAMS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True, help='folder with one sub-folder per match')
    ap.add_argument('--out-dir', default='outputs')
    ap.add_argument('--skip-preloss', action='store_true', help='do not run the pre-loss features')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    feats, log = extract_features(args.data_dir, os.path.join(args.out_dir, 'features.csv'), anchor='events')
    log.to_csv(os.path.join(args.out_dir, 'extraction_log.csv'), index=False)
    print(log.sum(numeric_only=True).to_string())
    if 'status' in log:
        print("skipped folders:", log.loc[log['status'].notna(), 'match'].tolist())
    sample = analysis_sample(feats)
    print(f"\nsequences, all teams: {len(feats):,}")
    print(f"Barcelona / Real Madrid losses: {int(feats['team_lost_possession'].isin(TEAMS).sum()):,}")
    print(f"analysis sample (ball reached own half): {len(sample):,}")
    print(sample.groupby(['team_lost_possession', 'outcome']).size().unstack().to_string())

    if not args.skip_preloss:
        print("\npre-loss features (robustness analysis 2) ...")
        extract_features(args.data_dir, os.path.join(args.out_dir, 'features_preloss.csv'), anchor='loss')


if __name__ == '__main__':
    main()
