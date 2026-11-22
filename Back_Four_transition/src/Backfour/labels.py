"""Outcome labels: failure (0) or success (1)."""
import ast

import numpy as np

from .config import BOX_DEPTH, BOX_Y_MAX, BOX_Y_MIN, DEF_THIRD_X, OWN_HALF_X
from .normalize import needs_flip, normalize_points

ON_BALL = {'Ball Receipt*', 'Carry', 'Dribble', 'Pass', 'Shot'}


def label_sequence(trk, f0, f1, period, def_side, hdl, seq, gained, seq_all=None):
    """(label, box_entry, shot, eng). label 0 = failure, 1 = success.

    Failure if, on any tracking frame from the loss (f0) to the 10th event (f1), the ball is inside the
    defending team's own penalty area, or the team that gained possession shoots, or the defending team
    scores an own goal. `eng` records when the ball first reached the own half and defensive third, the
    frame of the outcome (failures) and of an earlier regain of possession."""
    m = (trk['frame'] >= f0) & (trk['frame'] <= f1) & (trk['period'] == period)
    B, F = trk['ball'][m], trk['frame'][m]
    ok = ~np.isnan(B).any(axis=1)
    B, F = B[ok], F[ok]
    box_entry = False
    eng = {'min_ball_x': np.nan, 'entered_own_half': False, 'entered_def_third': False,
           'frame_entry_def_third': np.nan, 'frame_entry_own_half': np.nan,
           'entry_event_def_third': np.nan, 'entry_event_own_half': np.nan}
    if len(B):
        B = normalize_points(B, needs_flip(def_side, period, hdl))
        box_entry = bool(((B[:, 0] <= BOX_DEPTH) & (B[:, 1] >= BOX_Y_MIN) & (B[:, 1] <= BOX_Y_MAX)).any())
        eng['min_ball_x'] = round(float(B[:, 0].min()), 2)
        ev_frames = seq['skillcorner_frame'].astype(int).to_numpy()
        for tag, thr in (('def_third', DEF_THIRD_X), ('own_half', OWN_HALF_X)):
            inside = np.flatnonzero(B[:, 0] <= thr)
            if len(inside):
                fe = int(F[inside[0]])
                after = np.flatnonzero(ev_frames >= fe)
                eng[f'entered_{tag}'] = True
                eng[f'frame_entry_{tag}'] = fe
                eng[f'entry_event_{tag}'] = int(after[0]) + 1 if len(after) else len(ev_frames)
    f_box = int(F[(B[:, 0] <= BOX_DEPTH) & (B[:, 1] >= BOX_Y_MIN) & (B[:, 1] <= BOX_Y_MAX)][0]) if box_entry else np.inf
    sa = seq if seq_all is None or len(seq_all) == 0 else seq_all      # shots: look at all events
    shot_mask = (((sa['event_type'] == 'Shot') & (sa['team'] == gained))
                 | (sa['event_type'] == 'Own Goal Against'))
    shot = bool(shot_mask.any())
    f_shot = np.inf
    if shot:
        first = sa[shot_mask].iloc[0]
        if first['has_frame']:
            f_shot = int(first['skillcorner_frame'])
        else:   # shot without a frame: use the last framed event before it
            prev = sa[(sa['index'] < first['index']) & sa['has_frame']]
            f_shot = int(prev['skillcorner_frame'].max()) if len(prev) else f0
    eng['frame_outcome'] = min(f_box, f_shot)          # inf for successes
    # regain: the first event where possession is back with the defending team
    rg = sa[sa['possession_team'].notna() & (sa['possession_team'] != gained)]
    f_regain = np.inf
    if len(rg):
        e = rg.iloc[0]
        if e['has_frame']:
            f_regain = int(e['skillcorner_frame'])
        else:
            prev = sa[(sa['index'] < e['index']) & sa['has_frame']]
            f_regain = int(prev['skillcorner_frame'].max()) if len(prev) else f0
    eng['frame_regain'] = f_regain
    return (0 if (box_entry or shot) else 1), box_entry, shot, eng


def controlled_entry(seq_all, gained):
    """Alternative failure definition (sensitivity analysis only): True if the team that gained possession
    performed an on-ball action (receipt, carry, dribble, pass, shot) inside the penalty area it attacks.
    StatsBomb locations are oriented to the acting team (attacking towards x = 120): box = x >= 102 and
    18 <= y <= 62. Failed receptions ('Incomplete') do not count."""
    if seq_all is None or len(seq_all) == 0:
        return False
    for _, e in seq_all.iterrows():
        if e['team'] != gained or e['event_type'] not in ON_BALL:
            continue
        if e['event_type'] == 'Ball Receipt*' and e.get('outcome') == 'Incomplete':
            continue
        loc = e.get('location')
        if isinstance(loc, str):
            loc = ast.literal_eval(loc)
        if not isinstance(loc, (list, tuple)) or len(loc) < 2:
            continue
        if loc[0] >= 102 and 18 <= loc[1] <= 62:
            return True
    return False
