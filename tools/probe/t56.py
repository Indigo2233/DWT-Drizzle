# -*- coding: utf-8 -*-
"""跨夜抽样：T 到底有多大的变化（决定吞吐项该不该默认开）。"""
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
    n = len(fs_all)
    step = max(1, n // 24)
    pick = [Path(f) for f in fs_all[::step]][:24]
    print('共 %d 帧，抽 %d 帧' % (n, len(pick)))
    fr = pws.prepare_frames(pick, 'dbg2', 0, None)
    stars = pws.build_star_table(fr.frame(0), n_star=120)
    sh = drizzle.measure_shifts(fr, stars, ref=0, log=None)
    dx = np.asarray(sh['dx'], np.float64)
    dy = np.asarray(sh['dy'], np.float64)
    th = np.asarray(sh['theta'], np.float64)
    kk = np.asarray(sh.get('scale', np.ones(len(dx))), np.float64)
    mat = np.full((len(pick), len(stars)), np.nan)
    for i in range(len(pick)):
        mat[i] = pws.measure_flux_frame(fr.mm, i, fr.off, stars,
                                        th=th[i], kk=kk[i], dx=dx[i], dy=dy[i])
    with np.errstate(invalid='ignore', divide='ignore'):
        ref = np.nanmedian(np.where(mat > 0, mat, np.nan), axis=0)
        ratio = mat / ref[None]
    ratio = np.where(np.isfinite(ratio) & (ratio > 0), ratio, np.nan)
    med = np.nanmedian(ratio, axis=1)
    for i in range(len(pick)):
        name = os.path.basename(str(pick[i]))[:22]
        print('%s  有效 %3d  T=%.3f' % (name, int(np.isfinite(ratio[i]).sum()), med[i]))
    ok = np.isfinite(med)
    print('T 范围 %.3f~%.3f  中位 %.3f  相对离散 %.1f%%'
          % (np.nanmin(med), np.nanmax(med), np.nanmedian(med),
             100 * np.nanstd(med) / np.nanmedian(med)))


if __name__ == '__main__':
    main()
