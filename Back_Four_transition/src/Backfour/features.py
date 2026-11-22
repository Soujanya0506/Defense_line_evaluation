"""Frame-level indicators (Eq. 1-5) and their mean over the defending phase of a sequence."""
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

from .config import (BALL_RADIUS, BOX_DEPTH, BOX_Y_MAX, BOX_Y_MIN, FEATURE_HALF_WINDOW,
                     FEATURE_PRE_FRAMES, FIELD_W, FRAME_STEP, LAMBDA, PRESS_R, SPACE_EPS)
from .normalize import frame_state

# Zones of the space score (Table 1, Fig. 1), in PRIORITY order: a player counts only in the first
# zone that contains it. Each entry: (name, weight, test(point, ball)).
ZONES = [
    ('central_third', 0.35, lambda p, b: p[0] <= 35.0 and 19.0 <= p[1] <= 49.0),
    ('penalty_buffer', 0.30, lambda p, b: p[0] <= BOX_DEPTH + 5 and BOX_Y_MIN - 5 <= p[1] <= BOX_Y_MAX + 5),
    ('wing_pockets', 0.20, lambda p, b: p[0] <= 25.0 and (p[1] <= 10.0 or p[1] >= FIELD_W - 10.0)),
    ('ball_radius', 0.15, lambda p, b: np.hypot(p[0] - b[0], p[1] - b[1]) <= BALL_RADIUS),
]


def space_score(Dz, Az, ball):
    """Eq. 3. Each player counts once, in its highest-priority zone."""
    cnt = {name: [0, 0] for name, _, _ in ZONES}
    for k, pts in ((0, Dz), (1, Az)):
        for p in pts:
            for name, _, test in ZONES:
                if test(p, ball):
                    cnt[name][k] += 1
                    break
    return sum(w * (cnt[n][0] - cnt[n][1]) / (cnt[n][0] + cnt[n][1] + SPACE_EPS) for n, w, _ in ZONES)


def frame_features(D, A, ball):
    """Indicators at one normalized frame. The last line = the 4 defending outfield players with the
    smallest x; the three most advanced attackers = the 3 attacking outfield players with the smallest x."""
    if len(D) < 4 or len(A) < 3:
        return None
    B4 = D[np.argsort(D[:, 0])[:4]]
    A3 = A[np.argsort(A[:, 0])[:3]]
    try:
        hull_area = float(ConvexHull(B4).volume)          # 2-D "volume" = area
    except (QhullError, ValueError):
        hull_area = 0.0
    dmin = np.min(np.linalg.norm(A3[:, None, :] - B4[None, :, :], axis=2), axis=1)
    pda = float(dmin.mean())
    L = float(B4[:, 0].mean())
    return {'stretch_index': LAMBDA * hull_area + (1 - LAMBDA) * pda,            # Eq. 1
            'hull_area': hull_area, 'pda': pda,
            'pressure_index': int((dmin < PRESS_R).sum()),                       # Eq. 2, 0-3
            'space_score': space_score(B4, A3, ball),                            # Eq. 3
            'line_height_abs': L,                                                # Eq. 4
            'line_height_rel_ball': float(ball[0] - L)}                          # Eq. 5


def feature_frames(seq, f0, anchor='events'):
    """Candidate frames: +/- FEATURE_HALF_WINDOW frames around the loss event and each of the 10 events
    (anchor='events', main analysis), or the FEATURE_PRE_FRAMES frames up to the loss (anchor='loss')."""
    if anchor == 'loss':
        return set(range(f0 - FEATURE_PRE_FRAMES, f0 + 1, FRAME_STEP))
    frames = set()
    for f in [int(f0)] + seq['skillcorner_frame'].astype(int).tolist():
        frames.update(range(f - FEATURE_HALF_WINDOW, f + FEATURE_HALF_WINDOW + 1, FRAME_STEP))
    return frames


def sequence_features(trk, seq, def_side, period, hdl, f0, f_end=np.inf, anchor='events'):
    """Mean of each indicator over the candidate frames before f_end (the end of the defending phase).
    Returns (dict of means, number of frames used)."""
    rows = []
    for f in sorted(feature_frames(seq, f0, anchor)):
        if f >= f_end:
            continue
        s = frame_state(trk, f, def_side, period, hdl)
        if s is None:
            continue
        ff = frame_features(*s)
        if ff is not None:
            rows.append(ff)
    if not rows:
        return None, 0
    return pd.DataFrame(rows).mean().add_suffix('_mean').to_dict(), len(rows)
