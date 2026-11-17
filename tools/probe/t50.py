# -*- coding: utf-8 -*-
"""同一套测量，换成「原始帧 vs 注册帧」（1.020 尺度 + 180°，这才是堆叠时的情形）。"""
import sys
import glob
import os

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, 'core')
sys.path.insert(0, '.')
import matchref as M
import trireg
from t49 import fit_sim

REG = r'G:\HFG1-2\registered\Light_BIN-1_2752x2160_EXPOSURE-1200.00s_FILTER-O_mono'
CAL = r'G:\HFG1-2\calibrated\Light_BIN-1_3072x2240_EXPOSURE-1200.00s_FILTER-O_mono_DATE-2026-11-01'


def main():
    reg_path = glob.glob(os.path.join(REG, '2026-11-02_18-42*_r.xisf'))[0]
    reg = M._load(reg_path)
    sr = M.find_stars(reg, thr_pct=99.5, max_n=2000)
    for pat in ('2026-11-02_18-42*', '2026-11-04_23-12*', '2026-11-09_02-28*'):
        raw = M._load(glob.glob(os.path.join(CAL, pat))[0])
        ss = M.find_stars(raw, thr_pct=99.5, max_n=2000)
        tr = trireg.solve(ss, sr, max_pts=60)
        if not tr:
            print(pat, '三角失败')
            continue
        qg = (tr['scale'] * (tr['R'] @ ss.T)).T + tr['t']
        d, idx = cKDTree(sr).query(qg, distance_upper_bound=3.0)
        ok = np.isfinite(d)
        p, q = ss[ok], sr[idx[ok]]
        c = np.array(reg.shape) / 2.0
        r = np.linalg.norm(p - c, axis=1)
        s0, R0, t0 = fit_sim(p, q)
        res0 = np.linalg.norm((s0 * (R0 @ p.T)).T + t0 - q, axis=1)
        ang = np.degrees(np.arctan2(R0[1, 0], R0[0, 0]))
        print('%s  星对 %d  尺度 %.4f  角度 %.3f°' % (pat, ok.sum(), s0, ang))
        for lo, hi in [(0, 500), (500, 1000), (1000, 1500), (1500, 2200)]:
            m = (r >= lo) & (r < hi)
            if m.sum() < 5:
                continue
            print('    r %5d-%5d  n=%4d  残差均值 %.3f px  中位 %.3f'
                  % (lo, hi, m.sum(), res0[m].mean(), np.median(res0[m])))


if __name__ == '__main__':
    main()
