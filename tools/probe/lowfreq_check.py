# -*- coding: utf-8 -*-
"""低频残差在边缘是否更大：分带比较 128px 块中位相对全幅平面的偏离。"""
import sys
import os
import numpy as np
from scipy import ndimage

sys.path.insert(0, 'core')
sys.path.insert(0, '.')
import xisf_io

D = 'G:/HFG1-3/'


def lowfreq(path, block=128):
    a = np.asarray(xisf_io.read_xisf(path)[0].data, np.float64)
    h, w = a.shape
    gh, gw = h // block, w // block
    g = a[:gh * block, :gw * block].reshape(gh, block, gw, block)
    med = np.median(g, axis=(1, 3))
    med = ndimage.median_filter(med, 3)
    yy, xx = np.mgrid[0:gh, 0:gw]
    A = np.c_[np.ones(gh * gw), (yy - gh / 2).ravel() / gh, (xx - gw / 2).ravel() / gw]
    coef, *_ = np.linalg.lstsq(A, med.ravel(), rcond=None)
    plane = (A @ coef).reshape(gh, gw)
    resid = med - plane
    r = np.hypot(yy - gh / 2, xx - gw / 2) / max(gh, gw) * 2
    inner = r < 0.45
    edge = r > 0.9
    sig = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    print('%s  %dx%d 块 %dpx' % (os.path.basename(path), h, w, block))
    print('  平面峰谷 %.3g (%.2fσ)  残差 RMS %.3g (%.2fσ)'
          % (np.ptp(plane), np.ptp(plane) / sig, resid.std(), resid.std() / sig))
    print('  内圈 |残差| 中位 %.3g   外圈 %.3g   外/内 %.2f'
          % (np.median(np.abs(resid[inner])), np.median(np.abs(resid[edge])),
             np.median(np.abs(resid[edge])) / max(np.median(np.abs(resid[inner])), 1e-12)))


for f in ('PWS_x21_O_match_ast_ra.xisf', 'PWS_x21_H_match_ast_ra.xisf'):
    lowfreq(D + f)
