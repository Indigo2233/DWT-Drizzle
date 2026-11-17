# -*- coding: utf-8 -*-
"""用流水线自己的 measure_shifts 复现「配准失败」：3 帧小集合。"""
import sys
import glob
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, 'core')
sys.path.insert(0, '.')
import pws
import drizzle

CAL = r'G:\HFG1-3\calibrated'


def find(pat):
    h = glob.glob(os.path.join(CAL, '*', pat + '*.xisf'))
    return Path(h[0]) if h else None


def main():
    pats = ['2026-11-03_22-16-43_O', '2026-11-04_23-12-38_O', '2026-11-11_00-40-37_O']
    files = [find(p) for p in pats]
    files = [f for f in files if f]
    print('帧:', [f.name[:19] for f in files])

    fs = pws.prepare_frames(files, 'probe', 0, None)
    stars = pws.build_star_table(fs.frame(0), 120)
    print('星表 %d 颗' % len(stars))

    for ref in range(len(files)):
        print('--- 以 #%d 为参考 ---' % ref)
        try:
            sh = drizzle.measure_shifts(fs, stars, ref=ref, guess=None, log=print)
        except TypeError:
            sh = drizzle.measure_shifts(fs, stars, ref=ref, log=print)
        for k in ('dx', 'dy', 'theta', 'scale'):
            print('  %s = %s' % (k, np.round(np.asarray(sh[k], float), 4)))

if __name__ == '__main__':
    main()
