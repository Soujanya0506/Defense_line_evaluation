# backfour-transitions

Code for the paper **"Prediction-based evaluation of back-four defense with spatial control in soccer"**
(S. Dash, K. Ide, R. Umemoto, K. Amino and K. Fujii; *npj Artificial Intelligence*, under review).

The code evaluates the last line of four outfield players (the emergent back four) during negative
transitions, using synchronized tracking and event data:

1. **Sequences.** Open-play possession losses are found in StatsBomb event data synchronized with SkillCorner
   tracking (ETSY). For each loss, the 10 following synchronized events form a sequence; sequences interrupted
   by a stoppage or set piece are dropped unless a failure had already occurred.
2. **Labels.** A sequence is a failure if, on any tracking frame between the loss and the 10th event, the ball
   is inside the defending team's own penalty area, the team that gained possession shoots, or the defending
   team scores an own goal. Otherwise it is a success.
3. **Indicators.** On every frame within ±12 frames (±1.2 s at 10 Hz) of the loss and of each event, up to the
   end of the defending phase: stretch index, pressure index, space score, and absolute and relative line height
   (Eq. 1 to 5 of the paper). Coordinates are normalized so that the defending team defends the goal at x = 0.
4. **Analysis.** Possession losses by FC Barcelona and Real Madrid CF after which the ball reached the defending
   half: two-way ANOVA (Team × Outcome), success-failure mean differences with 95% CIs, team-specific classifiers
   (XGBoost, Random Forest, SVM) with SHAP, and four sensitivity analyses.

## Data

The tracking and event data are licensed from SkillCorner and StatsBomb and **cannot be shared**, so they are
not included. The code expects one folder per match:

```
DATA_DIR/
  <match folder>/
    events.csv      StatsBomb events with a `skillcorner_frame` column (frame matched by ETSY)
    tracking.json   SkillCorner tracking data (10 Hz)
    match.json      SkillCorner match metadata (teams, players, player roles, ball object id)
```

Columns of `events.csv` used: `index`, `period`, `team`, `possession_team`, `player`, `event_type`,
`event_type_2` (set-piece type), `location`, `outcome`, `skillcorner_frame`. Event types follow the StatsBomb
event data specification (StatsBomb Open Events v4.0.0).

## Installation

Python 3.10. The versions used for the paper are pinned in `requirements.txt`:

```
pip install -r requirements.txt
```

## Running

```
python scripts/01_extract_features.py --data-dir DATA_DIR --out-dir outputs   # sequences, labels, indicators (~20 min)
python scripts/02_analysis.py --out-dir outputs                               # all tables (~5 min)
python scripts/03_figures.py --data-dir DATA_DIR --out-dir outputs            # all figures
```

## Outputs and where they appear in the paper

| File (in `outputs/`) | Paper |
|---|---|
| `features.csv` | one row per sequence: label, indicators, sample flags |
| `table2_anova.csv` | Table 2, Supplementary Table S2 |
| `table3_mean_differences.csv` | Table 3, Supplementary Table S3 |
| `table4_class_weighted.csv` | Table 4, Supplementary Table S4 (A) |
| `table5_shap.csv` | Table 5 |
| `tableS1_descriptives_outliers.csv` | Supplementary Table S1 |
| `tableS4_balanced.csv`, `tableS4_stability.csv` | Supplementary Table S4 (B, C) |
| `tableS4_overfitting.csv` | Supplementary Table S5 |
| `robust_censoring.csv`, `robust_censoring_by_event.csv` | Supplementary Table S6 |
| `robust_preloss.csv`, `robust_preloss_auc.csv` | Supplementary Table S7 |
| `robust_outcome_B.csv`, `robust_outcome_B_auc.csv` | Supplementary Table S8 |
| `robust_match_grouped_cv.csv`, `robust_mixed_model.csv` | Supplementary Table S9 |
| `fig/zones.png`, `fig/feature_visual.png`, `fig/fig3_interaction.png` | Figures 1 to 3 |
| `fig/figS1_correlations.png` | Supplementary Figure S1 |

## Code layout

```
src/backfour/
  config.py      all settings: pitch, sequence length, feature windows, constants of Eq. 1-3, teams
  io.py          loading a match and building the tracking arrays
  normalize.py   which goal each team defends; normalization to x = 0
  events.py      event order, open-play turnovers, the 10-event sequences
  labels.py      failure / success (and the alternative definition used in Table S8)
  features.py    frame-level indicators (Eq. 1-5) and their mean over the defending phase
  pipeline.py    raw match files -> one row per sequence; the analysis sample
  stats.py       two-way ANOVA, mean differences, outliers and VIF, mixed models
  models.py      classifiers, bootstrap CIs, SHAP, balanced and match-grouped checks
  figures.py     Figures 1-3 and S1
scripts/         the three steps above
```

## Citation

If you use this code, please cite the paper (reference to be added on publication).

## License

MIT (see `LICENSE`).
