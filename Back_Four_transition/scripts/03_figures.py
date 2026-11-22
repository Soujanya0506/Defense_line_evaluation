"""Step 3: Figures 1-3 and Supplementary Figure S1.

    python scripts/03_figures.py --data-dir /path/to/matches --out-dir outputs

Needs outputs/features.csv (step 1) and outputs/table2_anova.csv (step 2); Figure 2 also reads the tracking
data of one match. Writes outputs/fig/zones.png, feature_visual.png, fig3_interaction.png and
figS1_correlations.png (300 dpi).
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from backfour import figures  # noqa: E402
from backfour.pipeline import analysis_sample  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True, help='folder with one sub-folder per match (for Figure 2)')
    ap.add_argument('--out-dir', default='outputs')
    ap.add_argument('--fig2-index', type=int, default=None, help='row of the analysis sample to draw in Figure 2')
    ap.add_argument('--fig2-frame', type=int, default=None, help='tracking frame to draw in Figure 2')
    args = ap.parse_args()
    fig_dir = os.path.join(args.out_dir, 'fig')
    os.makedirs(fig_dir, exist_ok=True)

    sample = analysis_sample(pd.read_csv(os.path.join(args.out_dir, 'features.csv')))
    table2 = pd.read_csv(os.path.join(args.out_dir, 'table2_anova.csv'))

    figures.fig_zones(os.path.join(fig_dir, 'zones.png'))
    info = figures.fig_feature_frame(sample, args.data_dir, os.path.join(fig_dir, 'feature_visual.png'),
                                     ex_index=args.fig2_index, ex_frame=args.fig2_frame)
    print("Figure 2:", info)
    figures.fig_interaction(sample, table2, os.path.join(fig_dir, 'fig3_interaction.png'))
    r_max = figures.fig_correlations(sample, os.path.join(fig_dir, 'figS1_correlations.png'))
    print(f"Figure S1: largest |r| between two indicators = {r_max:.2f}")
    print(f"figures written to {os.path.abspath(fig_dir)}")


if __name__ == '__main__':
    main()
