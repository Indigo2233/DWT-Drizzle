# -*- coding: utf-8 -*-
"""跨夜混合子集：逐帧看 θ / 匹配残差（scatter），找哪些帧没配上。"""
import numpy as np
import core.pws as pws
import core.drizzle as dz


def main():
    files, _ = pws.collect_frames(r'G:\HFG1-2\calibrated')
    groups = pws.group_frames_by_filter(files)
    h = groups['H']
    n1 = [f for f in h if '2026-10-05' in str(f)][:8]
    n2 = [f for f in h if '2026-11-01' in str(f)][:8]
    pick = n1 + n2
    fs = pws.prepare_frames(pick, 'mix', 0, None)
    stars = pws.build_star_table(fs.frame(len(n1) - 1), n_star=120)
    res = dz.measure_shifts(fs, stars, ref=len(n1) - 1, log=print)
    sc = res['scatter']
    print('%3s %10s %10s %9s %7s %s' % ('idx', 'dx', 'dy', 'theta', 'scatter', 'night'))
    for i in range(len(pick)):
        print('%3d %10.2f %10.2f %9.3f %7.3f %s' % (
            i, res['dx'][i], res['dy'][i], res['theta'][i],
            -1 if not np.isfinite(sc[i]) else sc[i],
            'n1' if i < len(n1) else 'n2'))


if __name__ == '__main__':
    main()
