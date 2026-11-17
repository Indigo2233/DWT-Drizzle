# -*- coding: utf-8 -*-
"""把用户两张去星通道拼成合成图，看瑕疵出现在哪。"""
import sys
import os
import numpy as np

sys.path.insert(0, 'core')
sys.path.insert(0, '.')
import xisf_io

D = 'G:/HFG1-3/'


def load(p):
    return np.asarray(xisf_io.read_xisf(p)[0].data, np.float64)


o = load(os.path.join(D, 'PWS_x21_O_match_ast_ra.xisf'))
h = load(os.path.join(D, 'PWS_x21_H_match_ast_ra.xisf'))


def norm(a, lo=0.002, hi=99.7):
    base = np.percentile(a, 50)
    top = np.percentile(a, hi)
    return np.clip((a - base) / max(top - base, 1e-9), 0, 1)


def asinh(x, k=8.0):
    return np.arcsinh(x * k) / np.arcsinh(k)


ro = asinh(norm(o))
rh = asinh(norm(h))
rgb = np.dstack([ro * 255, rh * 255, rh * 255]).astype(np.uint8)
np.save(r'C:\Windows\Temp\user_comp.npy', rgb[::4, ::4])
print('saved', rgb.shape)
