# -*- coding: utf-8 -*-
"""对一个 11 月帧：看 _rot_hist 给出的峰值、候选匹配数与最终解，定位误配。"""
import numpy as np
from scipy.spatial import cKDTree
import core.pws as pws
import core.drizzle as dz


def main():
    files, _ = pws.collect_frames(r'G:\HFG1-2\calibrated')
    groups = pws.group_frames_by_filter(files)
    h = groups['H']
    a = [f for f in h if '2026-10-30' in str(f)][0]
    b = [f for f in h if '2026-11-05' in str(f)][0]
    fs = pws.prepare_frames([a, b], 'pair', 0, None)
    ref, img = fs.frame(0), fs.frame(1)
    s0 = dz._detect_stars(ref)
    si = dz._detect_stars(img)
    print('星点', len(s0), len(si))
    info = dz._rot_hist(s0, si)
    print('rot_hist', None if info is None else 'th=%.3f votes=%d nb=%d kpk=%s' % (info[0], info[1], info[2], info[3]))
    # 手工在若干角度上试匹配：旋转后找最优平移，统计 1.5px 内命中
    tree = cKDTree(s0)
    for th_deg in (0.0, 175.0, 178.0, 179.0, 180.0, 181.0, 182.0):
        tr = np.radians(th_deg)
        rot = si @ np.array([[np.cos(tr), np.sin(tr)], [-np.sin(tr), np.cos(tr)]])
        best = None
        for j in range(len(rot)):
            u = s0[0] - rot[j]
            d, _ = tree.query(rot + u, k=1, distance_upper_bound=2.0)
            hits = int(np.sum(np.isfinite(d)))
            if best is None or hits > best[0]:
                best = (hits, u)
        hits, u = best
        print('  θ=%6.1f  2px 内命中 %3d/%d' % (th_deg, hits, len(si)))


if __name__ == '__main__':
    main()
