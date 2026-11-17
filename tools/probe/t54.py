# -*- coding: utf-8 -*-
"""逐帧吞吐探针：打印每帧的 FWHM、有效星数、流量中位与逐星归一后的 T。"""
import sys
import glob
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, 'core')
sys.path.insert(0, '.')
import pws


def main():
    cal = r'G:\HFG1-3\calibrated\Light_BIN-1_3072x2240_EXPOSURE-1200.00s_FILTER-H_mono_DATE-2026-11-01'
    fs = sorted(glob.glob(os.path.join(cal, '*.xisf')))
    print('共 %d 帧' % len(fs))
    pick = [Path(f) for f in fs[::max(1, len(fs) // 12)][:12]]
    fr = pws.prepare_frames(pick, 'dbg', 0, None)
    stars = pws.build_star_table(fr.frame(0), n_star=120)
    print('星表 %d 颗' % len(stars))
    rows = []
    for i in range(len(pick)):
        v, fl = pws.measure_fwhm_frame(fr.mm, i, fr.off, stars, with_flux=True)
        ok = np.isfinite(fl) & (fl > 0)
        rows.append((i, v, int(ok.sum()), float(np.nanmedian(fl[ok])) if ok.any() else float('nan'), fl))
    mat = np.full((len(rows), len(stars)), np.nan)
    for k, r in enumerate(rows):
        if r[4] is not None:
            mat[k] = r[4]
    with np.errstate(invalid='ignore', divide='ignore'):
        ref = np.nanmedian(np.where(mat > 0, mat, np.nan), axis=0)
        ratio = mat / ref[None]
    ratio = np.where(np.isfinite(ratio) & (ratio > 0), ratio, np.nan)
    for k, r in enumerate(rows):
        n_ok = int(np.isfinite(ratio[k]).sum())
        t = float(np.nanmedian(ratio[k])) if n_ok else float('nan')
        print('帧%2d FWHM %5.2f  有效星 %3d  流量中位 %8.4g  逐星归一 T %6.3f'
              % (r[0], r[1], n_ok, r[3], t))


if __name__ == '__main__':
    main()
