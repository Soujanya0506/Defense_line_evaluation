"""Figures 1-3 and Supplementary Figure S1."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Patch, Polygon, Rectangle
from scipy.spatial import ConvexHull

from .config import (BALL_RADIUS, BOX_DEPTH, BOX_Y_MAX, BOX_Y_MIN, FEATURES, FIELD_L, FIELD_W, NICE,
                     PRESS_R, TEAMS)
from .features import ZONES, frame_features
from .io import build_tracking, load_match
from .normalize import frame_state, home_defends_left

INK, LINE = '#0b0b0b', '0.55'
BLUE, ORANGE, YELLOW, GREEN = '#2a78d6', '#eb6834', '#eda100', '#1baf7a'
ZONE_COLORS = [BLUE, ORANGE, GREEN, YELLOW]                     # same order as ZONES (priority)
ZONE_NAMES = {'central_third': 'Central final third', 'penalty_buffer': 'Penalty box proximity',
              'wing_pockets': 'Wing pockets', 'ball_radius': 'Ball radius'}


def draw_pitch(ax, x_max=FIELD_L):
    """Pitch lines in metres, own goal at x = 0."""
    kw = dict(color=LINE, lw=1, zorder=3)
    ax.plot([0, x_max], [0, 0], **kw); ax.plot([0, x_max], [FIELD_W, FIELD_W], **kw)
    ax.plot([0, 0], [0, FIELD_W], **kw)
    if x_max >= FIELD_L:
        ax.plot([FIELD_L, FIELD_L], [0, FIELD_W], **kw)
    ax.plot([FIELD_L / 2] * 2, [0, FIELD_W], **kw)
    for x0, sgn in ((0, 1), (FIELD_L, -1)):
        if x0 > x_max:
            continue
        ax.add_patch(Rectangle((x0, BOX_Y_MIN), sgn * BOX_DEPTH, BOX_Y_MAX - BOX_Y_MIN, fill=False, ec=LINE, lw=1, zorder=3))
        ax.add_patch(Rectangle((x0, FIELD_W / 2 - 9.16), sgn * 5.5, 18.32, fill=False, ec=LINE, lw=1, zorder=3))
        ax.add_patch(Rectangle((x0, FIELD_W / 2 - 3.66), -sgn * 1.5, 7.32, fill=False, ec=LINE, lw=1.4, zorder=3))
        th = (-53, 53) if sgn == 1 else (127, 233)
        ax.add_patch(Arc((x0 + sgn * 11, FIELD_W / 2), 18.3, 18.3, theta1=th[0], theta2=th[1], color=LINE, lw=1, zorder=3))
    th = (90, 270) if x_max < FIELD_L else (0, 360)
    ax.add_patch(Arc((FIELD_L / 2, FIELD_W / 2), 18.3, 18.3, theta1=th[0], theta2=th[1], color=LINE, lw=1, zorder=3))


# ---------------------------------------------------------------- Figure 1
def fig_zones(out_path, ball_example=(44.0, 40.0), step=0.1):
    """Zones of the space score, drawn from ZONES itself so the figure always matches the code."""
    ball_ex = np.asarray(ball_example, float)
    x_max = FIELD_L / 2 + 2.5
    xs, ys = np.arange(0, x_max, step) + step / 2, np.arange(0, FIELD_W, step) + step / 2
    zone_idx = np.full((len(ys), len(xs)), -1)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            for k, (_, _, test) in enumerate(ZONES):
                if test((x, y), ball_ex):                          # first match = highest priority
                    zone_idx[j, i] = k
                    break
    shown = zone_idx.copy()                                         # thin gap where two zones meet
    for d in range(1, 4):
        for axis in (0, 1):
            for sh in (d, -d):
                nb = np.roll(zone_idx, sh, axis=axis)
                shown[(zone_idx >= 0) & (nb >= 0) & (nb != zone_idx) & (zone_idx > nb)] = -1

    fig, ax = plt.subplots(figsize=(6.2, 6.6))
    ax.imshow(np.ma.masked_less(shown, 0), cmap=ListedColormap(ZONE_COLORS), vmin=-0.5, vmax=3.5,
              alpha=0.30, origin='lower', extent=[0, x_max, 0, FIELD_W], interpolation='nearest', zorder=1)
    draw_pitch(ax, x_max=FIELD_L / 2)
    ax.scatter(*ball_ex, marker='*', s=160, c=YELLOW, edgecolor=INK, lw=0.8, zorder=5)
    w = {name: wt for name, wt, _ in ZONES}
    lab = dict(ha='center', va='center', fontsize=8.5, color=INK, zorder=6)
    ax.text(28.0, 34.0, f"Central\nfinal third\nw = {w['central_third']:.2f}", **lab)
    ax.annotate(f"Penalty box\nproximity\nw = {w['penalty_buffer']:.2f}", xy=(19.5, 54.5), xytext=(29.5, 54.0),
                arrowprops=dict(arrowstyle='-', color='0.4', lw=0.8), **lab)
    ax.text(12.5, 4.4, f"Wing pocket, w = {w['wing_pockets']:.2f}", **lab)
    ax.text(12.5, FIELD_W - 4.4, f"Wing pocket, w = {w['wing_pockets']:.2f}", **lab)
    ax.text(ball_ex[0], ball_ex[1] + BALL_RADIUS + 3.2, f"Ball radius ({BALL_RADIUS:g} m)\nw = {w['ball_radius']:.2f}", **lab)
    ax.set_xlim(-3, x_max); ax.set_ylim(-1, FIELD_W + 1); ax.set_aspect('equal')
    ax.set_xticks([0, BOX_DEPTH, BOX_DEPTH + 5, 25, 35, FIELD_L / 2])
    ax.set_xticklabels(['0', '16.5', '21.5', '25', '35', '52.5'], fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel('Distance from own goal line (m)', fontsize=9)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('0.7')
    ax.legend(handles=[Patch(fc=c, alpha=0.5, label=f"{i + 1}. {ZONE_NAMES[n]} (w = {wt:.2f})")
                       for i, ((n, wt, _), c) in enumerate(zip(ZONES, ZONE_COLORS))],
              loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8.5, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------- Figure 2
def split_roles(D, A):
    B4 = D[np.argsort(D[:, 0])[:4]]                      # D1 = deepest
    A3 = A[np.argsort(A[:, 0])[:3]]                      # A1 = most advanced
    oD = np.array([p for p in D if not np.isclose(B4, p).all(1).any()]).reshape(-1, 2)
    oA = np.array([p for p in A if not np.isclose(A3, p).all(1).any()]).reshape(-1, 2)
    return B4, A3, oD, oA


def frame_quality(D, A, ball, ff, r_target, min_gap=4.0):
    """Lower is better; inf = not usable for the figure (crowded, or ball outside 20-45 m)."""
    if ff is None:
        return np.inf
    B4, A3, _, _ = split_roles(D, A)
    P = np.vstack([B4, A3, np.asarray(ball)[None, :]])
    gaps = np.linalg.norm(P[:, None] - P[None, :], axis=2)[np.triu_indices(len(P), 1)]
    if gaps.min() < min_gap or not (20 <= ball[0] <= 45):
        return np.inf
    return abs(ff['line_height_rel_ball'] - r_target)


def _place_label(ax, x, y, text, obstacles, placed, d=2.4, char_w=0.8, **kw):
    """Label on the side of (x, y) furthest from markers and other labels."""
    w, h = char_w * len(text), 1.6
    opts = [(-d, 0, 'right', 'center', -d - w / 2, 0), (d, 0, 'left', 'center', d + w / 2, 0),
            (0, d, 'center', 'bottom', 0, d + h / 2), (0, -d, 'center', 'top', 0, -d - h / 2),
            (-d, d, 'right', 'bottom', -d - w / 2, d + h / 2), (d, d, 'left', 'bottom', d + w / 2, d + h / 2),
            (-d, -d, 'right', 'top', -d - w / 2, -d - h / 2), (d, -d, 'left', 'top', d + w / 2, -d - h / 2)]
    pts = np.array(list(obstacles) + list(placed)).reshape(-1, 2)
    best = max(opts, key=lambda o: np.min(np.linalg.norm(pts - [x + o[4], y + o[5]], axis=1)))
    ax.text(x + best[0], y + best[1], text, ha=best[2], va=best[3], fontsize=10, color=INK, zorder=9,
            bbox=dict(fc='white', ec='none', alpha=0.85, pad=0.4), **kw)
    placed.append([x + best[4], y + best[5]])


def draw_feature_frame(ax, D, A, ball, ff):
    """Normalized frame (own goal at x = 0) with the last line, the three most advanced attackers and
    the indicators; cropped to the part of the pitch that matters. Players are labelled by role only."""
    ball = np.asarray(ball, float)
    B4, A3, oD, oA = split_roles(D, A)
    key = np.vstack([B4, A3, ball[None, :]])
    x_max = float(np.clip(key[:, 0].max() + 12, FIELD_L / 2 + 5, FIELD_L))
    draw_pitch(ax)
    try:
        hv = ConvexHull(B4).vertices
        ax.add_patch(Polygon(B4[hv], closed=True, fc=BLUE, alpha=0.15, ec=BLUE, lw=1.2, zorder=2))
    except Exception:
        pass
    for p in B4:                                                         # 3 m pressure radius
        ax.add_patch(Circle(p, PRESS_R, fill=False, ec=BLUE, lw=0.9, ls=(0, (2, 2)), zorder=2))
    for a in A3:                                                         # PDA segments
        d = B4[np.argmin(np.linalg.norm(B4 - a, axis=1))]
        ax.plot([a[0], d[0]], [a[1], d[1]], color='0.3', lw=1.1, ls=':', zorder=3)
    L, R = ff['line_height_abs'], ff['line_height_rel_ball']
    ax.plot([L, L], [0, FIELD_W], color=BLUE, lw=1.5, ls='--', zorder=3)
    ax.text(L, FIELD_W + 1.0, 'L', ha='center', va='bottom', fontsize=11, color=INK, style='italic')
    ax.annotate('', xy=(ball[0], ball[1]), xytext=(L, ball[1]),
                arrowprops=dict(arrowstyle='->', color=INK, lw=1.3, shrinkA=0, shrinkB=7), zorder=4)
    vis = lambda P: P[P[:, 0] <= x_max] if len(P) else P
    if len(oD):
        ax.scatter(oD[:, 0], oD[:, 1], s=60, c='white', edgecolor=BLUE, lw=1.6, zorder=4)
    if len(oA):
        ax.scatter(oA[:, 0], oA[:, 1], s=60, c='white', edgecolor=ORANGE, lw=1.6, zorder=4)
    ax.scatter(B4[:, 0], B4[:, 1], s=190, c=BLUE, edgecolor='white', lw=1.8, zorder=5)
    ax.scatter(A3[:, 0], A3[:, 1], s=190, c=ORANGE, edgecolor='white', lw=1.8, marker='s', zorder=5)
    ax.scatter(*ball, marker='*', s=380, c=YELLOW, edgecolor=INK, lw=1, zorder=7)
    obstacles = np.vstack([key] + [P for P in (vis(oD), vis(oA)) if len(P)])
    placed = []
    for k, p in enumerate(B4):
        _place_label(ax, *p, f"D{k + 1}", obstacles, placed)
    for k, p in enumerate(A3):
        _place_label(ax, *p, f"A{k + 1}", obstacles, placed)
    _place_label(ax, (L + ball[0]) / 2, ball[1], 'R', obstacles, placed, d=1.6, style='italic')
    txt = (f"Line height L = {L:.1f} m\nRelative line height R = {R:.1f} m\n"
           f"Convex hull area = {ff['hull_area']:.1f} m$^2$\nPDA = {ff['pda']:.1f} m\n"
           f"Stretch index = {ff['stretch_index']:.1f}\nPressure index = {ff['pressure_index']}\n"
           f"Space score = {ff['space_score']:.3f}")
    bw, bh = 30.0, 19.0
    corners = {'lower right': (x_max - bw, 0), 'upper right': (x_max - bw, FIELD_W - bh),
               'lower left': (0, 0), 'upper left': (0, FIELD_W - bh)}

    def n_inside(c):
        x0, y0 = corners[c]
        return int(((obstacles[:, 0] >= x0) & (obstacles[:, 0] <= x0 + bw)
                    & (obstacles[:, 1] >= y0) & (obstacles[:, 1] <= y0 + bh)).sum())

    corner = min(corners, key=n_inside)
    ha, va = ('right' if 'right' in corner else 'left'), ('bottom' if 'lower' in corner else 'top')
    ax.text(x_max - 1 if ha == 'right' else 1, 1 if va == 'bottom' else FIELD_W - 1, txt, ha=ha, va=va,
            fontsize=9, color=INK, zorder=10, bbox=dict(fc='white', ec='0.7', lw=0.8, pad=4))
    ax.set_xlim(-3, x_max); ax.set_ylim(-1, FIELD_W + 5); ax.set_aspect('equal'); ax.axis('off')
    handles = [Line2D([], [], marker='o', ls='', ms=11, mfc=BLUE, mec='white', label='Last line (D1 to D4)'),
               Line2D([], [], marker='o', ls='', ms=7, mfc='white', mec=BLUE, mew=1.6, label='Other defending players'),
               Line2D([], [], marker='s', ls='', ms=11, mfc=ORANGE, mec='white', label='Three most advanced attackers (A1 to A3)'),
               Line2D([], [], marker='o', ls='', ms=7, mfc='white', mec=ORANGE, mew=1.6, label='Other attacking players'),
               Line2D([], [], marker='*', ls='', ms=15, mfc=YELLOW, mec=INK, label='Ball'),
               Line2D([], [], color=BLUE, ls='--', lw=1.5, label='Line height L'),
               Line2D([], [], color=INK, lw=1.3, marker='>', ms=5, label='Relative line height R'),
               Line2D([], [], color='0.3', ls=':', lw=1.1, label='Attacker to nearest last-line player (PDA)'),
               Line2D([], [], color=BLUE, ls=(0, (2, 2)), lw=0.9, label='3 m pressure radius')]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=9, frameon=False)


def fig_feature_frame(sample, data_dir, out_path, team='Barcelona', label=1, ex_index=None, ex_frame=None,
                      n_candidates=15, min_gap=4.0):
    """Figure 2. By default: among the successful Barcelona sequences closest to the median relative line
    height, the first one with a clear frame in its defending phase (ball 20-45 m from goal, key markers at
    least min_gap m apart, R closest to the sequence mean). Returns a dict describing the chosen frame."""
    if ex_index is not None:
        order = [ex_index]
    else:
        cand = sample[(sample['team_lost_possession'] == team) & (sample['label'] == label)]
        med = cand['line_height_rel_ball_mean'].median()
        order = (cand['line_height_rel_ball_mean'] - med).abs().sort_values().index[:n_candidates]
    cache, chosen = {}, None
    for idx in order:
        r = sample.loc[idx]
        if r['match_id'] not in cache:
            _events, _tracking, _match = load_match(data_dir, r['match_id'])
            _trk = build_tracking(_tracking, _match)
            cache[r['match_id']] = (_trk, home_defends_left(_trk))
        trk, hdl = cache[r['match_id']]
        side = 'h' if r['team_lost_possession'] == r['home_team'] else 'a'
        if ex_frame is not None:
            st = frame_state(trk, int(ex_frame), side, int(r['period']), hdl)
            chosen = (idx, int(ex_frame), st, frame_features(*st))
            break
        f_hi = int(min(r['frame_feature_end'], r['frame_last_event'] + 1))
        best = (np.inf, None, None, None)
        for f in range(int(r['frame_loss']), f_hi, 2):
            st = frame_state(trk, f, side, int(r['period']), hdl)
            if st is None:
                continue
            ff = frame_features(*st)
            q = frame_quality(*st, ff, r['line_height_rel_ball_mean'], min_gap)
            if q < best[0]:
                best = (q, f, st, ff)
        if best[1] is not None:
            chosen = (idx, best[1], best[2], best[3])
            break
    if chosen is None:
        raise RuntimeError("No clear frame found: lower min_gap or raise n_candidates.")
    idx, f_ex, (D, A, ball), ff = chosen
    fig, ax = plt.subplots(figsize=(7.0, 8.2))
    draw_feature_frame(ax, D, A, ball, ff)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    r = sample.loc[idx]
    return {'team': r['team_lost_possession'], 'outcome': 'success' if r['label'] == 1 else 'failure',
            'match_id': r['match_id'], 'frame': f_ex, 'sample_index': idx}


# ---------------------------------------------------------------- Figure 3
def fig_interaction(sample, table2, out_path):
    """Mean +/- SE of each indicator by team and outcome, with the Team x Outcome p-value of Table 2."""
    style = {'Barcelona': dict(color=BLUE, ls='-', marker='o'), 'Real Madrid': dict(color=ORANGE, ls='--', marker='s')}
    units = {'line_height_rel_ball_mean': 'm', 'line_height_abs_mean': 'm', 'stretch_index_mean': 'index',
             'pressure_index_mean': 'attackers within 3 m', 'space_score_mean': 'score'}
    fig, axs = plt.subplots(2, 3, figsize=(10, 6.2))
    for ax, f in zip(axs.flat, FEATURES):
        for team in TEAMS:
            g = (sample[sample['team_lost_possession'] == team]
                 .groupby('label')[f].agg(['mean', 'sem']).reindex([0, 1]))
            st = style[team]
            ax.errorbar([0, 1], g['mean'], yerr=g['sem'], color=st['color'], ls=st['ls'], marker=st['marker'],
                        ms=7, lw=2, capsize=4, label=team)
        p_int = table2.loc[(table2['feature'] == NICE[f]) & (table2['effect'] == 'Team * Outcome'), 'p'].iloc[0]
        ax.text(0.5, 1.02, f"Team × Outcome: p = {p_int:.3f}".replace('0.', '.'), transform=ax.transAxes,
                fontsize=8.5, ha='center', va='bottom', color='0.35')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Failure', 'Success'], fontsize=9); ax.set_xlim(-0.35, 1.35)
        ax.set_title(NICE[f], fontsize=10, pad=16)
        ax.set_ylabel(f"Mean ± SE ({units[f]})", fontsize=9)
        ax.tick_params(axis='y', labelsize=8.5)
        ax.grid(axis='y', color='0.9', lw=0.8); ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axs.flat[-1].axis('off')
    n = sample.groupby(['team_lost_possession', 'label']).size()
    handles, _ = axs.flat[0].get_legend_handles_labels()
    axs.flat[-1].legend(handles, [f"{t} (success n = {n[(t, 1)]}, failure n = {n[(t, 0)]})" for t in TEAMS],
                        loc='center', fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------- Figure S1
def fig_correlations(sample, out_path):
    """Pearson correlations between the five indicators. Returns the largest |r| between two indicators."""
    corr = sample[FEATURES].rename(columns=NICE).corr()
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=35, ha='right', fontsize=9)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=9)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', fontsize=8.5,
                    color='white' if abs(corr.iloc[i, j]) > 0.6 else INK)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Pearson r')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    off = corr.where(~np.eye(len(corr), dtype=bool)).abs()
    return float(off.max().max())
