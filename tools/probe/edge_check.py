# -*- coding: utf-8 -*-
"""量一下用户两张去星通道图的边缘噪声/电平。"""
import sys
import os
import numpy as np
from scipy import ndimage

sys.path.insert(0, 'core')
sys.path.insert(0, '.')
import xisf_io

D = 'G:/HFG1-3/'


def load(p):
    return np.asarray(xisf_io.read_xisf(p)[0].data, np.float64)


for f in ('PWS_x21_O_match_ast_ra.xisf', 'PWS_x21_H_match_ast_ra.xisf'):
    a = load(os.path.join(D, f))
    h, w = a.shape
    print('==', f, a.shape, 'median %.4g' % np.median(a))
    hp = a - ndimage.median_filter(a, 15)
    b = 32
    gh, gw = h // b, w // b
    blk = hp[:gh * b, :gw * b].reshape(gh, b, gw, b).std(axis=(1, 3))
    edge = np.zeros_like(blk, bool)
    edge[:3] = edge[-3:] = True
    edge[:, :3] = edge[:, -3:] = True
    ce, cc = np.median(blk[edge]), np.median(blk[~edge])
    print('  块噪声 中心 %.4g  边缘 %.4g  边缘/中心 %.2f' % (cc, ce, ce / cc))
    rows = []
    step = max(1, h // 8)
    for y0 in range(0, h, step):
        rows.append('%.4g' % np.median(a[y0:y0 + step]))
    print('  8 行带中位:', ' '.join(rows))
