# -*- coding: utf-8 -*-
"""先量清楚：边缘残差到底是「角度没估准」还是「系统性的场畸变」。

做法：取若干帧，用星点配准到参考帧，然后把残差按半径分箱；
再看「按半径加权重拟合」能不能把边缘残差压下去。
"""
import sys
import glob

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, 'core')
sys.path.insert(0, '.')
import matchref as M

REG = r'G:\HFG1-2\registered\Light_BIN-1_2752x2160_EXPOSURE-1200.00s_FILTER-O_mono'


def fit_sim(p, q, w=None):
    if w is None:
        w = np.ones(len(p))
    w = w / w.sum()
    pc = (p * w[:, None]).sum(0)
    qc = (q * w[:, None]).sum(0)
    P = p - pc
    Q = q - qc
    C = (Q * w[:, None]).T @ P
    U, S, Vt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(U @ Vt)) or 1.0
    R = U @ np.diag([1.0, d]) @ Vt
    var = (w * (P ** 2).sum(1)).sum()
    s = float(np.trace(np.diag(S) @ np.diag([1.0, d])) / max(var, 1e-12))
    t = qc - s * (R @ pc)
    return s, R, t


def main():
    files = sorted(glob.glob(REG + r'\*_r.xisf'))[:6]
    ref = M._load(files[0])
    sr = M.find_stars(ref, thr_pct=99.5, max_n=2000)
    print('参考帧星点', len(sr))
    for f in files[1:]:
        img = M._load(f)
        ss = M.find_stars(img, thr_pct=99.5, max_n=2000)
        # 先粗对齐（三角解），拿到初值
        import trireg
        tr = trireg.solve(ss, sr, max_pts=60)
        if not tr:
            print(f.split('\\')[-1][:30], '三角失败')
            continue
        q_guess = (tr['scale'] * (tr['R'] @ ss.T)).T + tr['t']
        d, idx = cKDTree(sr).query(q_guess, distance_upper_bound=3.0)
        ok = np.isfinite(d)
        p = ss[ok]
        q = sr[idx[ok]]
        c = np.array(ref.shape) / 2.0
        r = np.linalg.norm(p - c, axis=1)
        s0, R0, t0 = fit_sim(p, q)
        res0 = np.linalg.norm((s0 * (R0 @ p.T)).T + t0 - q, axis=1)
        w = r ** 2
        s1, R1, t1 = fit_sim(p, q, w)
        res1 = np.linalg.norm((s1 * (R1 @ p.T)).T + t1 - q, axis=1)
        ang0 = np.degrees(np.arctan2(R0[1, 0], R0[0, 0]))
        ang1 = np.degrees(np.arctan2(R1[1, 0], R1[0, 0]))
        bins = [(0, 500), (500, 1000), (1000, 1500), (1500, 2200)]
        line = f.split('\\')[-1][:26]
        print(f'{line}  星对 {ok.sum()}  角度 {ang0:.3f}->{ang1:.3f}°  尺度 {s0:.4f}->{s1:.4f}')
        for lo, hi in bins:
            m = (r >= lo) & (r < hi)
            if m.sum() < 5:
                continue
            print(f'    r {lo}-{hi}: n={m.sum():4d}  残差 普通 {res0[m].mean():.3f}  加权 {res1[m].mean():.3f} px')


if __name__ == '__main__':
    main()
