# -*- coding: utf-8 -*-
"""真实帧对的亚像素刚体拟合质量：残差随半径/方位角的分布（找场旋或畸变）。"""
import numpy as np
from scipy.spatial import cKDTree
import core.pws as pws
import core.drizzle as dz


def main():
    files, _ = pws.collect_frames(r'G:\HFG1-2\calibrated')
    groups = pws.group_frames_by_filter(files)
    h = groups['H'][:6]
    fs = pws.prepare_frames(h, 'Hprobe', 0, None)
    ref = fs.frame(0)
    s0 = dz._detect_stars(ref)
    print('参考帧星点', len(s0))
    cy, cx = np.asarray(fs.shape) / 2.0
    for i in (1, 5):
        img = fs.frame(i)
        si = dz._detect_stars(img)
        info = dz._rot_hist(s0, si)
        th = float(info[0]) if info is not None else 0.0
        tr = np.radians(th)
        rot = si @ np.array([[np.cos(tr), np.sin(tr)], [-np.sin(tr), np.cos(tr)]])
        tree = cKDTree(s0)
        best = None
        for j in range(len(rot)):
            u = s0[0] - rot[j]
            d, _ = tree.query(rot + u, k=1, distance_upper_bound=4.0)
            hits = int(np.sum(np.isfinite(d)))
            if best is None or hits > best[0]:
                best = (hits, u)
        hits, u = best
        d, idx = tree.query(rot + u, k=1, distance_upper_bound=6.0)
        ok = np.isfinite(d)
        q = s0[idx[ok]]
        p = si[ok]
        # 亚像素重心
        pf, pr = [], []
        for (px, py), (qx, qy) in zip(p, q):
            ix, iy = int(round(px)), int(round(py))
            jx, jy = int(round(qx)), int(round(qy))
            if ix < 6 or iy < 6 or ix >= img.shape[1] - 6 or iy >= img.shape[0] - 6:
                continue
            c1 = dz._centroid(img, ix, iy, 6)
            c2 = dz._centroid(ref, jx, jy, 6)
            if not (np.all(np.isfinite(c1[:2])) and np.all(np.isfinite(c2[:2]))):
                continue
            pf.append(c1[:2]); pr.append(c2[:2])
        pf = np.asarray(pf); pr = np.asarray(pr)
        th2, k2, ux, uy, rms = dz._fit_sim(pf, pr)
        R = np.array([[np.cos(np.radians(th2)), -np.sin(np.radians(th2))],
                      [np.sin(np.radians(th2)), np.cos(np.radians(th2))]])
        pred = k2 * (pf @ R.T) + np.array([ux, uy])
        res = pr - pred
        rr = np.hypot(pr[:, 0] - cx, pr[:, 1] - cy)
        print('帧 %d: n=%d θ=%.4f° k=%.6f rms=%.4f px' % (i, len(pf), th2, k2, rms))
        for lo, hi in ((0, 400), (400, 800), (800, 1200), (1200, 1800)):
            m = (rr >= lo) & (rr < hi)
            if m.sum() >= 3:
                print('   r %4d-%4d n=%3d  残差均值 (%+.3f,%+.3f)  RMS %.3f  切向分量 %+.3f'
                      % (lo, hi, m.sum(), res[m, 0].mean(), res[m, 1].mean(),
                         np.sqrt((res[m] ** 2).sum(1).mean()),
                         (res[m, 0] * -(pr[m, 1] - cy) + res[m, 1] * (pr[m, 0] - cx)).sum()
                         / max(np.sum(rr[m]), 1e-9)))


if __name__ == '__main__':
    main()
