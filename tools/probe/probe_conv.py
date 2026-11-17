# -*- coding: utf-8 -*-
"""确定 trireg 解 → 流水线 (theta, k, u) 的换算：穷举符号组合看哪个命中率高。"""
import sys
import glob
import os
import itertools
import numpy as np
from pathlib import Path

sys.path.insert(0, 'core')
sys.path.insert(0, '.')
import pws
import drizzle
import trireg

CAL = r'G:\HFG1-3\calibrated'


def find(pat):
    h = glob.glob(os.path.join(CAL, '*', pat + '*.xisf'))
    return Path(h[0]) if h else None


def main():
    seed_p = find('2026-11-03_22-16-43_O')
    bad_p = find('2026-11-04_23-12-38_O')
    fs = pws.prepare_frames([seed_p, bad_p], 'probe', 0, None)
    s_ref = drizzle._detect_stars(fs.frame(0))
    s_i = drizzle._detect_stars(fs.frame(1))
    print('星点 ref=%d src=%d' % (len(s_ref), len(s_i)))

    tr = trireg.solve(np.asarray(s_i, float), np.asarray(s_ref, float), max_pts=90)
    print('trireg: n=%d scale=%.4f theta=%.2f rms=%.2f t=%s'
          % (tr['n_in'], tr['scale'], tr['theta'], tr['rms'], np.round(tr['t'], 1)))

    best = []
    for ksign, tsign, usign in itertools.product((1, -1), repeat=3):
        k = tr['scale'] ** ksign
        th = tr['theta'] * tsign
        u = tr['t'] * usign
        ok, hit, nt = drizzle._verify_all(s_ref, s_i, th, k, u)
        best.append((hit / max(nt, 1), ksign, tsign, usign, hit, nt, ok))
    best.sort(reverse=True)
    for b in best[:4]:
        print('  k%s θ%s u%s → %d/%d = %.0f%% ok=%s'
              % ('^+1' if b[1] > 0 else '^-1', '+θ' if b[2] > 0 else '-θ',
                 '+t' if b[3] > 0 else '-t', b[4], b[5], 100 * b[0], b[6]))


if __name__ == '__main__':
    main()
