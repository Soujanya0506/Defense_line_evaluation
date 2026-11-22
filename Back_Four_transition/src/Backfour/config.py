"""Settings of the analysis (Methods, Eq. 1-5, Table 1).

All coordinates are in metres on a 105 x 68 m pitch. After normalization the defending
team always defends the goal at x = 0. Tracking data are sampled at 10 Hz.
"""

# ---- pitch ----
FIELD_L, FIELD_W = 105.0, 68.0
BOX_DEPTH = 16.5
BOX_Y_MIN = (FIELD_W - 40.32) / 2          # 13.84
BOX_Y_MAX = (FIELD_W + 40.32) / 2          # 54.16

# ---- sequences ----
SEQ_LEN = 10                  # synchronized events after the possession loss
EXCLUDE_LOSS_IN_BOX = True    # a loss inside the own penalty area is dropped

# ---- frames used for the features (10 Hz) ----
FEATURE_HALF_WINDOW = 12      # +/- 12 frames (1.2 s) around the loss event and each of the 10 events
FEATURE_PRE_FRAMES = 12       # robustness analysis: only the 12 frames up to the loss
FRAME_STEP = 1

# ---- feature constants ----
LAMBDA = 0.5                  # Eq. 1: stretch = 0.5 * convex-hull area + 0.5 * PDA
PRESS_R = 3.0                 # Eq. 2: pressure radius (m)
SPACE_EPS = 1.0               # Eq. 3: epsilon
BALL_RADIUS = 3.0             # Table 1: ball zone radius (m)

# ---- analysis ----
OWN_HALF_X = 52.5             # analysis sample: the ball reached x <= 52.5 within the 10 events
DEF_THIRD_X = 35.0            # recorded for description only
TEAMS = ['Barcelona', 'Real Madrid']
FEATURES = ['line_height_rel_ball_mean', 'line_height_abs_mean', 'stretch_index_mean',
            'pressure_index_mean', 'space_score_mean']
NICE = {'line_height_rel_ball_mean': 'Line height (relative)', 'line_height_abs_mean': 'Line height (absolute)',
        'stretch_index_mean': 'Stretch index', 'pressure_index_mean': 'Pressure index',
        'space_score_mean': 'Space score'}
ALPHA_BONF = 0.05 / len(FEATURES)          # five ANOVAs -> .01
SEED = 42
N_DRAWS = 20                               # balanced 50/50 draws (supplementary)
N_BOOT = 2000                              # bootstrap resamples for the test ROC-AUC CI

# ---- names and event vocabularies ----
TEAM_NAMES = {   # SkillCorner team name -> StatsBomb team name
    'UD Almería': 'Almería', 'Real Sociedad': 'Real Sociedad', 'Athletic Club de Bilbao': 'Athletic Club',
    'Villarreal CF': 'Villarreal', 'RC Celta de Vigo': 'Celta Vigo', 'Getafe CF': 'Getafe',
    'UD Las Palmas': 'Las Palmas', 'Sevilla FC': 'Sevilla', 'Cadiz CF': 'Cádiz',
    'Atlético Madrid': 'Atlético Madrid', 'RCD Mallorca': 'Mallorca', 'Valencia CF': 'Valencia',
    'CA Osasuna': 'Osasuna', 'Girona FC': 'Girona', 'Real Betis Balompié': 'Real Betis',
    'FC Barcelona': 'Barcelona', 'Deportivo Alavés': 'Deportivo Alavés', 'Granada CF': 'Granada',
    'Rayo Vallecano': 'Rayo Vallecano', 'Real Madrid CF': 'Real Madrid'}

# Not possession actions: ignored when looking for a change of possession.
NON_POSSESSION = {'Starting XI', 'Half Start', 'Half End', 'Tactical Shift', 'Substitution',
                  '50/50', 'Offside', 'Foul Committed', 'Foul Won', 'Injury Stoppage',
                  'Referee Ball-Drop', 'Error'}
# Stoppages: a sequence that meets one of these before the 10th event is interrupted.
RESTART_TYPE = {'Substitution', 'Half Start', 'Half End', 'Offside', 'Foul Committed',
                'Foul Won', 'Injury Stoppage', 'Referee Ball-Drop'}
# Set pieces are recorded in `event_type_2`; the tag can be glued to another flag
# (e.g. 'TrueFree Kick'), so a tag counts as a set piece if it CONTAINS one of these names.
RESTART_TYPE2 = ('Goal Kick', 'Throw-in', 'Free Kick', 'Corner', 'Kick Off')
