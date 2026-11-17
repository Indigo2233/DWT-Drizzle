# -*- coding: utf-8 -*-
"""吞吐探针：配准后同一批星的流量，逐帧打印（定位病态帧）。"""
import sys
import glob
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, 'core')
sys.path.insert(0, '.')
import pws
import drizzle


def main():
    cal = r'G:\HFG1-3\calibrated\Light_BIN-1_3072x2240_EXPOSURE-1200.00s_FILTER-H_mono_DATE-2026-11-01'
    fs_all = sorted(glob.glob(os.path.join(cal, '*.xisf')))
    pick = [Path(f) for f in fs_all[::max(1, len(fs_all) // 12)][:12]]
    fr = pws.prepare_frames(pick, 'dbg', 0, None)
    stars = pws.build_star_table(fr.frame(0), n_star=120)
    print('星表 %d 颗' % len(stars))
    sh = drizzle.measure_shifts(fr, stars, ref=0, log=None)
    dx = np.asarray(sh['dx'], np.float64)
    dy = np.asarray(sh['dy'], np.float64)
    th = np.asarray(sh['theta'], np.float64)
    kk = np.asarray(sh.get('scale', np.ones(len(dx))), np.float64)
    rows = []
    for i in range(len(pick)):
        for sgn in (1.0,):
            fl = pws.measure_flux_frame(fr.mm, i, fr.off, stars, th=th[i], kk=kk[i],
                                        dx=sgn * dx[i], dy=sgn * dy[i])
            ok = np.isfinite(fl) & (fl > 0)
            rows.append((i, sgn, int(ok.sum()),
                         float(np.median(fl[ok])) if ok.any() else float('nan')))
    for r in rows:
        print('帧%2d sign%+d 有效星 %3d 流量中位 %.4g' % r)


if __name__ == '__main__':
    main()
