# -*- coding: utf-8 -*-
"""诊断：H 通道逐帧刚体配准的残差是否随半径增长（场旋中心不一致 → 切向拖影）。"""
import sys
import numpy as np
import core.pws as pws
import core.drizzle as dz


def main():
    files, src = pws.collect_frames(r'G:\HFG1-2\calibrated')
    groups = pws.group_frames_by_filter(files)
    h = groups['H'][:6]
    fs = pws.prepare_frames(h, 'Hprobe', 0, None)
    print('shape', fs.shape)
    s0 = dz._detect_stars(fs.frame(0))
    print('det ref', len(s0))
    cy, cx = np.asarray(fs.shape) / 2.0
    for i in (1, 3, 5):
        img = fs.frame(i)
        si = dz._detect_stars(img)
        info = dz._rot_hist(s0, si)
        print('frame', i, 'det', len(si), 'rot_hist', 'None' if info is None else round(float(info[0]), 4))
        # 手工：用最优粗角构造 k=1 的旋转，再用 KD 匹配 40px 内 + SVD
        from scipy.spatial import cKDTree
        th = float(info[0]) if info is not None else 0.0
        tr = np.radians(th)
        rot = si @ np.array([[np.cos(tr), np.sin(tr)], [-np.sin(tr), np.cos(tr)]])
        tree = cKDTree(s0)
        # 粗平移：取若干候选的最小中位距离
        best = None
        for j in range(0, len(rot), max(1, len(rot) // 200)):
            for k in range(0, len(s0), max(1, len(s0) // 200)):
                u = s0[k] - rot[j]
                d, _ = tree.query(rot + u, k=1, distance_upper_bound=4.0)
                hits = int(np.sum(np.isfinite(d)))
                if best is None or hits > best[0]:
                    best = (hits, u)
        hits, u = best
        print('  coarse hits', hits, 'u', np.round(u, 2))
        d, idx = tree.query(rot + u, k=1, distance_upper_bound=6.0)
        ok = np.isfinite(d)
        q = s0[idx[ok]]
        p = si[ok]  # 匹配到的输入坐标（与 rot 同序）
        # 以两帧各自中心为原点做刚体拟合
        pc = p - np.array([cx, cy])
        qc = q - np.array([cx, cy])
        A = np.c_[pc, np.ones(len(pc))]
        solx, *_ = np.linalg.lstsq(A, qc[:, 0], rcond=None)
        soly, *_ = np.linalg.lstsq(A, qc[:, 1], rcond=None)
        # solx = [a, -b, tx]，soly = [b, a, ty] 形式：用相似变换最小二乘
        th2, k2, ux, uy, rms = dz._fit_sim(p, q)
        r = np.linalg.norm(qc, axis=1)
        # 用拟合结果预测，看残差是否随半径增长
        tr2 = np.radians(th2)
        R = k2 * np.array([[np.cos(tr2), np.sin(tr2)], [-np.sin(tr2), np.cos(tr2)]])
        pred = p @ R.T + np.array([ux, uy])
        res = np.linalg.norm(pred - q, axis=1)
        print('  fit th %.4f k %.5f u (%.2f,%.2f) rms %.3f n %d' % (th2, k2, ux, uy, rms, len(p)))
        for lo, hi in ((0, 300), (300, 700), (700, 1100), (1100, 1600)):
            m = (r >= lo) & (r < hi)
            if m.sum() > 3:
                print('   r %4d-%4d  n=%4d  rms %.3f  mean_dx %+.3f  mean_dy %+.3f'
                      % (lo, hi, m.sum(), np.sqrt((res[m] ** 2).mean()),
                         (pred[m, 0] - q[m, 0]).mean(), (pred[m, 1] - q[m, 1]).mean()))


if __name__ == '__main__':
    main()
