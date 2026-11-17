# -*- coding: utf-8 -*-
"""验证 11-01 之后的帧到底是「真换了尺度」还是「兜底假解」。

对 10-30 帧与 11-05 帧，直接在两组星点上做 KD 匹配：
  固定 θ=180°，分别试 k=1.000 与 k=1.020，看 2.5px 内命中数。
"""
import sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, 'core')
import pws
from drizzle import _detect_stars


def main():
    files, _ = pws.collect_frames(r'G:\HFG1-2\calibrated')
    h = pws.group_frames_by_filter(files)['H']
    a = [f for f in h if '2026-10-30' in str(f)][0]
    b = [f for f in h if '2026-11-05' in str(f)][0]
    fs = pws.prepare_frames([a, b], 'cmp', 0, None)
    sa = np.asarray(_detect_stars(np.asarray(fs.frame(0))), np.float64)
    sb = np.asarray(_detect_stars(np.asarray(fs.frame(1))), np.float64)
    print('星数 A=%d B=%d' % (len(sa), len(sb)))
    th = np.radians(180.0)
    for k in (1.000, 1.005, 1.010, 1.015, 1.020, 1.025, 1.030):
        ct, st = np.cos(th), np.sin(th)
        r = k * (sb @ np.array([[ct, st], [-st, ct]]))
        # 用两帧重心的差当平移
        u = sa.mean(0) - r.mean(0)
        d, _ = cKDTree(sa).query(r + u, k=1, distance_upper_bound=2.5)
        hit = int(np.sum(np.isfinite(d)))
        print('k=%.3f  命中 %3d/%3d  (%.0f%%)' % (k, hit, len(sb), 100.0 * hit / max(len(sb), 1)))


if __name__ == '__main__':
    main()
