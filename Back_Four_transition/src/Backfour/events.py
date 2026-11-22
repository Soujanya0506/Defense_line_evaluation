"""Event order, possession turnovers and the 10-event sequences."""
import pandas as pd

from .config import NON_POSSESSION, RESTART_TYPE, RESTART_TYPE2, SEQ_LEN


def is_set_piece_tag(tag):
    """True if an event_type_2 value is (or contains) a set-piece name."""
    s = str(tag)
    return any(k in s for k in RESTART_TYPE2)


def sync_events(events, trk):
    """Events with a matched tracking frame (ETSY), in StatsBomb order."""
    ev = events.dropna(subset=['skillcorner_frame']).copy()
    ev['skillcorner_frame'] = ev['skillcorner_frame'].astype(int)
    rows = ev['skillcorner_frame'].map(trk['row_of'])
    ev = ev[rows.notna()].copy()
    ev['period'] = trk['period'][rows[rows.notna()].astype(int).values]
    if 'event_type_2' not in ev.columns:
        ev['event_type_2'] = pd.NA
    return ev.sort_values(['period', 'index']).reset_index(drop=True)


def order_all_events(events, ev):
    """All events (with or without a tracking frame) in StatsBomb order, with a `has_frame` flag.
    Used to detect stoppages, set pieces and shots that have no frame; positions always come from `ev`."""
    a = events.copy()
    if 'event_type_2' not in a.columns:
        a['event_type_2'] = pd.NA
    a['has_frame'] = a['index'].isin(ev['index'])
    return a.sort_values(['period', 'index']).reset_index(drop=True)


def find_turnovers(ev, ev_all):
    """Open-play possession turnovers among events with a frame.
    A change of possession is a set-piece restart (and dropped) if the new team's events between the
    last framed event of the old possession and this one include a set piece, framed or not."""
    pe = ev[~ev['event_type'].isin(NON_POSSESSION)].copy()
    pe['prev_index'] = pe['index'].shift(1)
    pe['prev_team'] = pe['possession_team'].shift(1)
    pe['prev_type'] = pe['event_type'].shift(1)
    pe['prev_player'] = pe['player'].shift(1) if 'player' in pe.columns else pd.NA
    pe['prev_period'] = pe['period'].shift(1)
    turns = pe[(pe['possession_team'] != pe['prev_team'])
               & pe['possession_team'].notna() & pe['prev_team'].notna()
               & (pe['period'] == pe['prev_period'])]

    def starts_with_set_piece(t):
        gap = ev_all[(ev_all['period'] == t['period']) & (ev_all['index'] > t['prev_index'])
                     & (ev_all['index'] <= t['index']) & (ev_all['possession_team'] == t['possession_team'])]
        return bool(gap['event_type_2'].map(is_set_piece_tag).any())

    is_restart = turns.apply(starts_with_set_piece, axis=1).astype(bool) if len(turns) else pd.Series(dtype=bool)
    return turns[~is_restart], int(is_restart.sum())


def next_events(ev_all, t):
    """(seq, complete, seq_all) for the turnover t.
    Walks through all events after the turnover (framed or not) and counts only framed ones.
    complete = True : SEQ_LEN framed events reached with no stoppage or set piece in between.
    complete = False: a stoppage or set piece came first; seq = framed events before it (None if none).
    seq     = the framed events (positions and feature windows)
    seq_all = every event up to the last one used (shots, own goals and regains, framed or not)."""
    after = ev_all[(ev_all['period'] == t['period']) & (ev_all['index'] > t['index'])].head(SEQ_LEN * 6)
    seq, seq_all = [], []
    for _, e in after.iterrows():
        if e['event_type'] in RESTART_TYPE or is_set_piece_tag(e['event_type_2']):
            return (pd.DataFrame(seq) if seq else None), False, pd.DataFrame(seq_all)
        seq_all.append(e)
        if e['has_frame']:
            seq.append(e)
            if len(seq) == SEQ_LEN:
                return pd.DataFrame(seq), True, pd.DataFrame(seq_all)
    return (pd.DataFrame(seq) if seq else None), False, pd.DataFrame(seq_all)
