# -*- coding: utf-8 -*-
"""F1(bgdeg0) vs Z2(bgdeg1)：同一窗口裁切、同一拉伸，并排 + 单出 xisf。

用法: python crop_compare.py
"""
import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, r'G:\DWT\DWT-drizzle')
sys.path.insert(0, r'G:\DWT\DWT-drizzle\core')
from xisf_io import read_xisf, write_xisf, XISFImage

F1 = r'G:\HFG1\lab\F1_246_orig\PWS_F1_246_orig.xisf'
Z2 = r'G:\HFG1\lab\Z2_246_bg1\PWS_Z2_246_bg1.xisf'
OUT = r'G:\HFG1\lab\crop_bg'
os.makedirs(OUT, exist_ok=True)


def load(p):
    return np.asarray(read_xisf(p)[0].data, np.float32)


def crop(img, cy, cx, half):
    """按中心分数坐标裁 2*half 见方，越界则夹回。"""
    h, w = img.shape[:2]
    y0 = int(round(cy * h)) - half
    x0 = int(round(cx * w)) - half
    y0 = max(0, min(y0, h - 2 * half))
    x0 = max(0, min(x0, w - 2 * half))
    return img[y0:y0 + 2 * half, x0:x0 + 2 * half].copy(), (y0, x0)


def stretch(a, lo, hi, g=0.45):
    return (np.clip((a - lo) / (hi - lo), 0, 1) ** g * 255).astype(np.uint8)


def sheet(a, b, path, scale=1, g=0.45, sig_mult=12.0):
    lo = np.median(np.concatenate([a.ravel(), b.ravel()]))
    sig = 1.4826 * np.median(np.abs(np.concatenate([a[::4, ::4].ravel(), b[::4, ::4].ravel()]) - lo))
    ia = Image.fromarray(stretch(a, lo - 1.5 * sig, lo + sig_mult * sig, g))
    ib = Image.fromarray(stretch(b, lo - 1.5 * sig, lo + sig_mult * sig, g))
    if scale != 1:
        ia = ia.resize((ia.width * scale, ia.height * scale), Image.NEAREST)
        ib = ib.resize((ib.width * scale, ib.height * scale), Image.NEAREST)
    w = ia.width + ib.width + 8
    out = Image.new('L', (w, ia.height), 0)
    out.paste(ia, (0, 0))
    out.paste(ib, (ia.width + 8, 0))
    out.save(path)
    print(path, out.size, 'lo %.4e sig %.3e' % (lo, sig))


def dump_xisf(img, path, tag):
    im = XISFImage()
    im.data = np.ascontiguousarray(img, np.float32)
    im.geometry = (img.shape[1], img.shape[0], 1)
    im.bounds = (0.0, 1.0)
    im.fits_keywords = [{'name': 'HISTORY', 'value': tag, 'comment': 'crop for bgdeg A/B'}]
    write_xisf(path, im)
    print(path, img.shape)


def main():
    f1, z2 = load(F1), load(Z2)
    print('F1', f1.shape, 'Z2', z2.shape)
    # (中心分数 y, x, 半宽, 名称) —— 星云区、星点区
    jobs = [
        (0.695, 0.622, 512, 'neb'),
        (0.235, 0.330, 512, 'star'),
    ]
    for cy, cx, half, name in jobs:
        a, oa = crop(f1, cy, cx, half)
        b, ob = crop(z2, cy, cx, half)
        print(name, 'F1 origin', oa, 'Z2 origin', ob)
        sheet(a, b, os.path.join(OUT, 'cmp_%s.png' % name), 1)
        dump_xisf(a, os.path.join(OUT, 'F1_bg0_%s.xisf' % name), 'bgdeg=0 %s y0,x0=%d,%d' % (name, oa[0], oa[1]))
        dump_xisf(b, os.path.join(OUT, 'Z2_bg1_%s.xisf' % name), 'bgdeg=1 %s y0,x0=%d,%d' % (name, ob[0], ob[1]))


if __name__ == '__main__':
    main()
