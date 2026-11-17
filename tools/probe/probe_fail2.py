# -*- coding: utf-8 -*-
"""诊断配准失败的帧：星点数、三角配准内点、以及「全星验证命中率」。"""
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
    hits = glob.glob(os.path.join(CAL, '*', pat + '*.xisf'))
    return hits[0] if hits else None


def hit_rate(ss, sr, k, th, u, rad=2.5):
    """把 src 星点按 p_ref = (1/k)·R(-θ)·p + u 投到参考帧，数命中率。"""
    c, s = np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th))
    p = np.asarray(ss, float)
    q = np.c_[(p[:, 1] * c + p[:, 0] * s) / k + u[0],
              (-p[:, 1] * s + p[:, 0] * c) / k + u[1]]
    tree = cKDTree(np.asarray(sr, float))
    d, _ = tree.query(q, distance_upper_bound=rad)
    return float(np.isfinite(d).mean())


def probe(ref, path, tag):
    sr = M.find_stars(ref)
    img = M._load(path)
    ss = M.find_stars(img)
    out, info = M.align(img, ref, log=lambda *a: None)
    ok = out is not None
    print('%-22s 星点 ref=%d src=%d  align=%s' % (tag, len(sr), len(ss), 'OK' if ok else 'FAIL'))
    if info:
        for kk in ('n_pair', 'k', 'theta', 'rms', 'cover', 'error'):
            if kk in info:
                print('    %-7s %s' % (kk, info[kk]))
        if 'k' in info and 'u' in info:
            print('    全星命中率 %.1f%%' % (100 * hit_rate(ss, sr, info['k'], info['theta'], info['u'])))


if __name__ == '__main__':
    ref = M._load(find('2026-11-03_22-16-43_O'))
    print('参考帧自身的星点数 %d' % len(M.find_stars(ref)))
    for f in ['2026-11-04_23-12-38_O', '2026-11-05_22-25-27_O', '2026-11-11_00-40-37_O']:
        p = find(f)
        if p:
            probe(ref, p, f[:19])
