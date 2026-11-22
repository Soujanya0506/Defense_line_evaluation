"""From raw match files to one row per defensive sequence."""
import os

import pandas as pd

from .config import BOX_DEPTH, BOX_Y_MAX, BOX_Y_MIN, EXCLUDE_LOSS_IN_BOX, RESTART_TYPE2, TEAM_NAMES, TEAMS
from .events import find_turnovers, is_set_piece_tag, next_events, order_all_events, sync_events
from .features import sequence_features
from .io import build_tracking, list_matches, load_match
from .labels import controlled_entry, label_sequence
from .normalize import frame_state, home_defends_left

LOG_KEYS = ('turnovers', 'setpiece_restart', 'unknown_team', 'no_frame', 'few_players', 'short_sequence',
            'loss_in_box', 'no_features', 'kept', 'glued_setpiece_tags', 'kept_failed_before_stoppage',
            'events_without_frame')


def process_match(data_dir, match_dir, anchor='events'):
    """All defensive sequences of one match. Returns (rows, log of kept and excluded turnovers)."""
    log = {'match': match_dir, **{k: 0 for k in LOG_KEYS}}
    loaded = load_match(data_dir, match_dir)
    if loaded is None:
        log['status'] = 'missing files'
        return [], log
    events, tracking, match = loaded
    home, away = TEAM_NAMES[match['home_team']['name']], TEAM_NAMES[match['away_team']['name']]
    trk = build_tracking(tracking, match)
    hdl = home_defends_left(trk)
    ev = sync_events(events, trk)
    ev_all = order_all_events(events, ev)
    log['events_without_frame'] = int((~ev_all['has_frame']).sum())
    tags = ev['event_type_2'].dropna().astype(str)
    log['glued_setpiece_tags'] = int((tags.map(is_set_piece_tag) & ~tags.isin(RESTART_TYPE2)).sum())
    turns, n_restart = find_turnovers(ev, ev_all)
    log['turnovers'], log['setpiece_restart'] = len(turns) + n_restart, n_restart

    rows = []
    for _, t in turns.iterrows():
        lost, gained, period = t['prev_team'], t['possession_team'], int(t['period'])
        if lost not in (home, away):
            log['unknown_team'] += 1; continue
        def_side = 'h' if lost == home else 'a'
        f0 = int(t['skillcorner_frame'])
        st = frame_state(trk, f0, def_side, period, hdl)
        if st is None:
            log['no_frame'] += 1; continue
        D0, _, ball0 = st
        if len(D0) < 4:                                   # at least four outfield players for the last line
            log['few_players'] += 1; continue
        seq, complete, seq_all = next_events(ev_all, t)
        if seq is None:
            log['short_sequence'] += 1; continue
        f1 = int(seq['skillcorner_frame'].max())
        label, box_entry, shot, eng = label_sequence(trk, f0, f1, period, def_side, hdl, seq, gained, seq_all)
        if not complete and label == 1:                   # stopped early, outcome unknown: dropped
            log['short_sequence'] += 1; continue
        loss_in_box = bool(ball0[0] <= BOX_DEPTH and BOX_Y_MIN <= ball0[1] <= BOX_Y_MAX)
        if EXCLUDE_LOSS_IN_BOX and loss_in_box:
            log['loss_in_box'] += 1; continue
        # the features cover the defending phase only: up to the outcome (failure),
        # or the 10th event / an earlier regain (success)
        f_end = min(eng['frame_outcome'], eng['frame_regain'], f1 + 1)
        feats, n_frames = sequence_features(trk, seq, def_side, period, hdl, f0, f_end=f_end, anchor=anchor)
        if feats is None:
            log['no_features'] += 1; continue
        if not complete:
            log['kept_failed_before_stoppage'] += 1
        ctrl = controlled_entry(seq_all, gained)          # alternative definition (sensitivity analysis)
        label_B = 0 if (ctrl or shot) else 1
        valid_B = bool(complete or label_B == 0)
        rows.append({'match_id': match_dir, 'transition_idx': t['index'], 'label': label,
                     'home_team': home, 'away_team': away,
                     'team_lost_possession': lost, 'team_gained_possession': gained, 'period': period,
                     'player_lost_possession': t['prev_player'],
                     'loss_event_type': t['prev_type'], 'turnover_event_type': t['event_type'],
                     'frame_loss': f0, 'frame_last_event': f1,
                     'n_events': len(seq), 'n_events_incl_unframed': len(seq_all), 'complete_10_events': complete,
                     'ball_x_at_loss': round(float(ball0[0]), 2), 'ball_y_at_loss': round(float(ball0[1]), 2),
                     'box_entry': box_entry, 'shot_against': shot, 'loss_in_box': loss_in_box,
                     'controlled_entry': ctrl, 'label_B': label_B, 'valid_B': valid_B, **eng,
                     'frame_feature_end': f_end, 'n_feature_frames': n_frames, **feats})
        log['kept'] += 1
    return rows, log


def extract_features(data_dir, out_csv, anchor='events', progress=True):
    """Runs process_match on every match folder and writes one row per sequence to out_csv.
    Returns (features, log). anchor='loss' gives the robustness analysis with pre-loss features."""
    matches = list_matches(data_dir, exclude=('combined_outputs', os.path.basename(os.path.dirname(out_csv))))
    if progress:
        try:
            from tqdm import tqdm
            matches = tqdm(matches)
        except ImportError:
            pass
    rows_all, logs = [], []
    for d in matches:
        rows, log = process_match(data_dir, d, anchor=anchor)
        rows_all += rows
        logs.append(log)
    out = pd.DataFrame(rows_all)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out, pd.DataFrame(logs)


def analysis_sample(features):
    """Possession losses by Barcelona or Real Madrid after which the ball reached the own half."""
    two = features[features['team_lost_possession'].isin(TEAMS)]
    sample = two[two['entered_own_half'].astype(bool)].copy()
    sample['outcome'] = sample['label'].map({0: 'Failure', 1: 'Success'})
    return sample
