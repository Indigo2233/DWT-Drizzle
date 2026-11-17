# -*- coding: utf-8 -*-
"""对比不同 R 下限的成品：星点锐度、噪声、细节保留。

用法：compare_rfloor.py 输出目录1 输出目录2 …
每个目录里要有 PWS_*.xisf 与 *_R.npy。
"""
import sys
import glob
import os

import numpy as np
from scipy import ndimage

sys.path.insert(0, 'core')
sys.path.insert(0, '.')
import xisf_io


def load(d):
    f = [p for p in glob.glob(os.path.join(d, 'PWS_*.xisf')) if '_match' not in p][0]
    img = np.asarray(xisf_io.read_xisf(f)[0].data, np.float64)
    rf = glob.glob(os.path.join(d, '*_R.npy'))
    R = np.load(rf[0]) if rf else None
    return img, R, os.path.basename(f)


def stats(img, masks):
    small = img[::2, ::2]
    hf = small - ndimage.median_filter(small, 3)
    out = {}
    for name, m in masks.items():
        m = m[::2, ::2]
        if m.shape != hf.shape:
            continue
        if m.sum() < 1000:
            out[name] = None
            continue
        v = hf[m]
        v = v[np.isfinite(v)]
        out[name] = (float(np.std(v)), int(m.sum()))
    return out


def main():
    dirs = sys.argv[1:]
    ref_img, ref_R, _ = load(dirs[0])
    masks = {'细节': ref_R > 0.6, '星云': (ref_R >= 0.05) & (ref_R <= 0.25), '天空': ref_R < 0.02}
    print('掩膜（取自 %s 的 R 场）：' % dirs[0]
          + '  '.join('%s %d px' % (k, int(v.sum() // 4)) for k, v in masks.items()))
    for d in dirs:
        img, R, name = load(d)
        if R is None:
            print('%s: 没有 R 场' % d)
            continue
        s = stats(img, masks)
        parts = []
        for k in ('细节', '星云', '天空'):
            if s[k]:
                parts.append('%s 高频 σ %.3e（%d px）' % (k, s[k][0], s[k][1]))
        print('%-22s %s  天空中位 %.4g' % (name, '  '.join(parts), float(np.median(img))))


if __name__ == '__main__':
    main()
