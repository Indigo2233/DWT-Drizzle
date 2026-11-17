# -*- coding: utf-8 -*-
"""两晚各取一帧：拟合相似变换的 k/θ 与残差随半径的分布（找跨夜尺度差）。"""
import numpy as np
import core.pws as pws
import core.drizzle as dz


def fit_pair(ref, img, label):
    s0 = dz._detect_stars(ref)
    si = dz._detect_stars(img)
    info = dz._rot_hist(s0, si)
    th = float(info[0]) if info is not None else 0.0
    tr = np.radians(th)
    rot = si @ np.array([[np.cos(tr), np.sin(tr)], [-np.sin(tr), np.cos(tr)]])
    from scipy.spatial import cKDTree
    tree = cKDTree(s0)
    best = None
    for j in range(len(rot)):
        u = s0[0] - rot[j]
        d, _ = tree.query(rot + u, k=1, distance_upper_bound=4.0)
        hits = int(np.sum(np.isfinite(d)))
        if best is None or hits > best[0]:
            best = (hits, u)
    hits, u = best
    d, idx = tree.query(rot + u, k=1, distance_upper_bound=8.0)
    ok = np.isfinite(d)
    q, p = s0[idx[ok]], si[ok]
    pf, pr = [], []
    for (px, py), (qx, qy) in zip(p, q):
        ix, iy = int(round(px)), int(round(py))
        jx, jy = int(round(qx)), int(round(qy))
        if ix < 8 or iy < 8 or ix >= img.shape[1] - 8 or iy >= img.shape[0] - 8:
            continue
        c1 = dz._centroid(img, ix, iy, 8)
        c2 = dz._centroid(ref, jx, jy, 8)
        if not (np.all(np.isfinite(c1[:2])) and np.all(np.isfinite(c2[:2]))):
            continue
        pf.append(c1[:2]); pr.append(c2[:2])
    pf, pr = np.asarray(pf), np.asarray(pr)
    th2, k2, ux, uy, rms = dz._fit_sim(pf, pr)
    R = np.array([[np.cos(np.radians(th2)), -np.sin(np.radians(th2))],
                  [np.sin(np.radians(th2)), np.cos(np.radians(th2))]])
    res = pr - (k2 * (pf @ R.T) + np.array([ux, uy]))
    rr = np.hypot(pr[:, 0] - ref.shape[1] / 2, pr[:, 1] - ref.shape[0] / 2)
    print('%s: n=%d θ=%.4f° k=%.6f rms=%.4f' % (label, len(pf), th2, k2, rms))
    for lo, hi in ((0, 400), (400, 800), (800, 1200), (1200, 1800)):
        m = (rr >= lo) & (rr < hi)
        if m.sum() >= 3:
            print('   r %4d-%4d n=%3d rms %.3f  径向分量 %+.3f' % (
                lo, hi, m.sum(), np.sqrt((res[m] ** 2).sum(1).mean()),
                (res[m, 0] * (pr[m, 0] - ref.shape[1] / 2) + res[m, 1] * (pr[m, 1] - ref.shape[0] / 2)).sum()
                / max(np.sum(rr[m]), 1e-9)))


def main():
    files, _ = pws.collect_frames(r'G:\HFG1-2\calibrated')
    groups = pws.group_frames_by_filter(files)
    h = groups['H']
    n1 = [f for f in h if '2026-10-05' in str(f)]
    n2 = [f for f in h if '2026-11-01' in str(f)]
    print('晚1 帧数', len(n1), '晚2 帧数', len(n2))
    fs = pws.prepare_frames([n1[0], n1[1], n2[0]], 'cross', 0, None)
    fit_pair(fs.frame(0), fs.frame(1), '同晚 n1[0]→n1[1]')
    fit_pair(fs.frame(0), fs.frame(2), '跨夜 n1[0]→n2[0]')


if __name__ == '__main__':
    main()
