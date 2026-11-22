"""Which goal each team defends, and normalization so that the defending team defends x = 0."""
import numpy as np

from .config import FIELD_L, FIELD_W


def home_defends_left(trk):
    """{period: True if the HOME team defends raw x = 0}, from each team's goalkeepers' median x."""
    out = {}
    for per in np.unique(trk['period']):
        m = trk['period'] == per
        hx = trk['h'][m][:, trk['gk']['h'], 0] if trk['gk']['h'] else np.array([np.nan])
        ax = trk['a'][m][:, trk['gk']['a'], 0] if trk['gk']['a'] else np.array([np.nan])
        out[int(per)] = bool(np.nanmedian(hx) < np.nanmedian(ax))
    return out


def needs_flip(def_side, period, hdl):
    defends_left = hdl[period] if def_side == 'h' else not hdl[period]
    return not defends_left


def normalize_points(pts, flip):
    """The only place where coordinates are mirrored (single 180-degree rotation)."""
    if not flip:
        return pts
    out = pts.copy()
    out[..., 0] = FIELD_L - out[..., 0]
    out[..., 1] = FIELD_W - out[..., 1]
    return out


def frame_state(trk, frame, def_side, period, hdl):
    """Normalized (defending outfield players, attacking outfield players, ball) at one frame,
    or None if the frame or the ball is missing. Goalkeepers (substitutes included) are excluded."""
    i = trk['row_of'].get(int(frame))
    if i is None or trk['period'][i] != period or np.isnan(trk['ball'][i]).any():
        return None
    flip = needs_flip(def_side, period, hdl)
    att_side = 'a' if def_side == 'h' else 'h'

    def outfield(side):
        P = trk[side][i]
        keep = np.ones(23, dtype=bool)
        keep[trk['gk'][side]] = False
        P = P[keep]
        return P[~np.isnan(P).any(axis=1)]

    D = normalize_points(outfield(def_side), flip)
    A = normalize_points(outfield(att_side), flip)
    ball = normalize_points(trk['ball'][i][None, :], flip)[0]
    return D, A, ball
