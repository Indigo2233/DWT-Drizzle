# -*- coding: utf-8 -*-
"""用 matchref 的正确约定复核命中率：p_ref = (1/k)·R(-θ)·p_src + u。"""
import sys
import os
import glob
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, 'core')
sys.path.insert(0, '.')
import matchref as M

CAL = r'G:\HFG1-3\calibrated'


def find(pat):
    h = glob.glob(os.path.join(CAL, '*', pat + '*.xisf'))
    return h[0] if h else None


def hit(ss, sr, k, th, u, rad=2.5):
    c, s = np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th))
    p = np.asarray(ss, float)
    x, y = p[:, 0], p[:, 1]
    q = np.c_[(x * c + y * s) / k + u[0], (-x * s + y * c) / k + u[1]]
    d, _ = cKDTree(np.asarray(sr, float)).query(q, distance_upper_bound=rad)
    return float(np.isfinite(d).mean())


if __name__ == '__main__':
    ref = M._load(find('2026-11-03_22-16-43_O'))
    sr = M.find_stars(ref)
    for f in ['2026-11-04_23-12-38_O', '2026-11-05_22-25-27_O',
              '2026-11-11_00-40-37_O', '2026-11-06_22-51-42_O']:
        p = find(f)
        if not p:
            continue
        img = M._load(p)
        ss = M.find_stars(img)
        out, info = M.align(img, ref, log=lambda *a: None)
        if not info or 'k' not in info:
            print(f[:19], 'align 失败', info)
            continue
        print('%s  n=%d rms=%.2f k=%.4f th=%.1f  命中率 %.1f%%'
              % (f[:19], info['n_pair'], info['rms'], info['k'], info['theta'],
                 100 * hit(ss, sr, info['k'], info['theta'], info['u'])))
