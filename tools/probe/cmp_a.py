# -*- coding: utf-8 -*-
"""A 值扫描的目视对比：同一块细节区，三种 A 并排。"""
import sys
import glob
import os

import numpy as np
from scipy import ndimage

sys.path.insert(0, 'core')
sys.path.insert(0, '.')
import xisf_io

RUNS = [('A=20 default', r'G:\HFG1\lab\W60_fp2'),
        ('A=14', r'G:\HFG1\lab\W60_A14'),
        ('A=10', r'G:\HFG1\lab\W60_A10')]


def main():
    R = np.load(glob.glob(os.path.join(RUNS[0][1], '*_R.npy'))[0])
    ys, xs = np.nonzero(R > 0.6)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    S, Z = 420, 2
    tiles = []
    for name, d in RUNS:
        f = [p for p in glob.glob(os.path.join(d, 'PWS_*.xisf')) if '_match' not in p][0]
        img = np.asarray(xisf_io.read_xisf(f)[0].data, np.float64)
        t = img[cy - S // 2:cy + S // 2, cx - S // 2:cx + S // 2]
        lo = float(np.median(t))
        hi = float(np.percentile(t, 99.7))
        z = np.arcsinh(np.clip((t - lo) / max(hi - lo, 1e-9) * 6, 0, None)) / np.arcsinh(6.0)
        im = (np.clip(z, 0, 1) * 255).astype(np.uint8)
        tiles.append(np.repeat(np.repeat(im, Z, 0), Z, 1))
    sep = np.full((tiles[0].shape[0], 8), 255, np.uint8)
    row = tiles[0]
    for t in tiles[1:]:
        row = np.hstack([row, sep, t])
    np.save(r'C:\Windows\Temp\A_sweep.npy', row)
    print('center', cy, cx, 'tile', tiles[0].shape)


if __name__ == '__main__':
    main()
