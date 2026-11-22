"""Loading one match: StatsBomb events (synchronized with ETSY) and SkillCorner tracking."""
import json
import os

import numpy as np
import pandas as pd

from .config import FIELD_L, FIELD_W

MATCH_FILES = ('events.csv', 'tracking.json', 'match.json')


def list_matches(data_dir, exclude=('combined_outputs',)):
    """Match folders inside data_dir (one folder per match)."""
    return sorted(d for d in os.listdir(data_dir)
                  if os.path.isdir(os.path.join(data_dir, d)) and d not in exclude)


def load_match(data_dir, match_dir):
    """(events, tracking, match), or None if a file is missing."""
    path = os.path.join(data_dir, match_dir)
    files = [os.path.join(path, f) for f in MATCH_FILES]
    if not all(os.path.exists(p) for p in files):
        return None
    events = pd.read_csv(files[0])
    with open(files[1]) as fh:
        tracking = json.load(fh)
    with open(files[2], encoding='utf-8') as fh:
        match = json.load(fh)
    return events, tracking, match


def build_tracking(tracking, match):
    """Raw frames -> arrays indexed by frame number. Player slot k = k-th player of that team in match.json."""
    home_id = match['home_team']['id']
    slot, gk, count = {}, {'h': [], 'a': []}, {'h': 0, 'a': 0}
    for p in match['players']:
        side = 'h' if p['team_id'] == home_id else 'a'
        k = count[side]
        if k >= 23:
            continue
        count[side] += 1
        slot[p['trackable_object']] = (side, k)
        if p['player_role']['name'] == 'Goalkeeper':
            gk[side].append(k)
    ball_obj = match['ball']['trackable_object']

    frames, periods, rows_h, rows_a, rows_b = [], [], [], [], []
    for fr in tracking:
        if fr.get('period') is None or fr.get('frame') is None or not fr.get('data'):
            continue
        h, a, b = np.full((23, 2), np.nan), np.full((23, 2), np.nan), np.full(2, np.nan)
        for o in fr['data']:
            x, y = o['x'] + FIELD_L / 2, -o['y'] + FIELD_W / 2
            if o['trackable_object'] == ball_obj:
                b[:] = (x, y)
            elif o['trackable_object'] in slot:
                side, k = slot[o['trackable_object']]
                (h if side == 'h' else a)[k] = (x, y)
        frames.append(int(fr['frame'])); periods.append(int(fr['period']))
        rows_h.append(h); rows_a.append(a); rows_b.append(b)

    frames = np.array(frames)
    order = np.argsort(frames)
    trk = {'frame': frames[order], 'period': np.array(periods)[order],
           'h': np.stack(rows_h)[order], 'a': np.stack(rows_a)[order],
           'ball': np.stack(rows_b)[order], 'gk': gk}
    trk['row_of'] = {f: i for i, f in enumerate(trk['frame'])}
    return trk
