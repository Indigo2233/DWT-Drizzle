# -*- coding: utf-8 -*-
"""真实帧对：残差随半径的变化，以及二阶径向项能不能吃掉它。

用阈值法在整个画幅上取 500+ 颗星（不用 _detect_stars 的 80 颗上限），
先做相似变换粗配，再把匹配对按半径分箱看残差；最后网格搜 d2 看残差能降多少。
"""
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, 'core')
import pws
import probe_pa as P
import drizzle as D


def main():
    files, _ = pws.collect_frames(r'G:\HFG1-2\calibrated')
    h = pws.group_frames_by_filter(files)['H']
    pa = [f for f in h if '2026-10-30' in str(f)][0]
    pb = [f for f in h if '2026-11-05' in str(f)][0]
    fs = pws.prepare_frames([pa, pb], 'cmp', 0, None)
    a = np.asarray(fs.frame(0), np.float32)
    b = np.asarray(fs.frame(1), np.float32)
    sa = P.stars(a)
    sb = P.stars(b)
    print('星数 A=%d B=%d' % (len(sa), len(sb)))
    # 一阶粗配：先按亮星重心 + KD 最近邻的直方图法（用 D._rot_hist 同思路简化）
    # 这里直接用 A 的星去 B 的星里找平移（两帧同指向，角度差很小）
    ta = cKDTree(sb[:, :2])
    best = None
    for dx in range(-40, 41, 2):
        for dy in range(-40, 41, 2):
            d, _ = ta.query(sa[:, :2] + [dx, dy], k=1, distance_upper_bound=3.0)
            hit = int(np.sum(np.isfinite(d)))
            if best is None or hit > best[0]:
                best = (hit, dx, dy)
    hit, dx0, dy0 = best
    print('粗配 ±40 搜索: dx=%d dy=%d 命中 %d' % (dx0, dy0, hit))
    q = sa[:, :2] + [dx0, dy0]
    d, ii = ta.query(q, k=1, distance_upper_bound=3.0)
    ok = np.isfinite(d)
    print('匹配 %d 对' % ok.sum())
    pp = sa[ok, :2]
    qq = sb[ii[ok], :2]
    th, k, ux, uy, rms = D._fit_sim(pp, qq)
    print('相似解 θ=%.3f° k=%.4f 残差 %.3fpx' % (th, k, rms))
    c = (a.shape[1] / 2.0, a.shape[0] / 2.0)
    rc = 0.5 * min(a.shape)
    r = np.hypot(pp[:, 0] - c[0], pp[:, 1] - c[1])
    res = np.hypot(*(qq - (k * (pp @ np.array([[np.cos(np.radians(th)), np.sin(np.radians(th))],
                                                [-np.sin(np.radians(th)), np.cos(np.radians(th))]]).T
                             ) + [ux, uy])).T)
    for lo, hi in ((0, 400), (400, 800), (800, 1200), (1200, 1600)):
        m = (r >= lo) & (r < hi)
        if m.sum() >= 5:
            print('  r %4d-%4d n=%3d 残差 中位 %.3fpx' % (lo, hi, m.sum(), np.median(res[m])))
    th2, k2, ux2, uy2, rms2, d2 = D._fit_sim_dist(pp, qq, c, rc, -0.03, 0.03, 61)
    print('带畸变 d2=%+.4f 残差 %.3f → %.3fpx' % (d2, rms, rms2))


if __name__ == '__main__':
    main()
