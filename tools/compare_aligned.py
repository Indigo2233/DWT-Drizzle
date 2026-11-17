# -*- coding: utf-8 -*-
"""O9 / F1(bgdeg0) / Z2(bgdeg1) / master 在 O9 裁切画幅（2008x2458）上的并排对比。

输出：
  G:\HFG1\lab\crop_bg\aligned_sheet.png   2x2 全幅（0.5x 下采样）
  G:\HFG1\lab\crop_bg\aligned_neb.png     1:1 裁切 800²，四方并排
  G:\HFG1\lab\crop_bg\aligned_diff.png    Z2-F1 差值（x8 放大对比）
"""
import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, r'G:\DWT\DWT-drizzle')
sys.path.insert(0, r'G:\DWT\DWT-drizzle\core')
from xisf_io import read_xisf

OUT = r'G:\HFG1\lab\crop_bg'
MASTER = r'G:\HFG1-2\master\masterLight_BIN-1_2752x2160_EXPOSURE-1200.00s_FILTER-O_mono_(1)_autocrop.xisf'
ITEMS = [
    ('master', MASTER),
    ('O9', r'G:\HFG1\drz_o9\o9_aligned.xisf\o9_aligned.xisf'),
    ('F1 bg0', r'G:\HFG1\lab\F1_246_orig\f1_aligned.xisf'),
    ('Z2 bg1', r'G:\HFG1\lab\Z2_246_bg1\z2_aligned.xisf'),
]


def load(p):
    return np.asarray(read_xisf(p)[0].data, np.float32)


def stretch(a, lo, hi, g=0.45):
    return (np.clip((a - lo) / (hi - lo), 0, 1) ** g * 255).astype(np.uint8)


def grid(imgs, path, g=0.45, scale=1.0, sig_lo=-1.5, sig_hi=14.0):
    allv = np.concatenate([a[::4, ::4].ravel() for _, a, _ in imgs])
    med = float(np.median(allv))
    sig = 1.4826 * float(np.median(np.abs(allv - med)))
    lo, hi = med + sig_lo * sig, med + sig_hi * sig
    tiles = []
    for name, a, sub in imgs:
        c = a[sub[0]:sub[2], sub[1]:sub[3]]
        im = Image.fromarray(stretch(c, lo, hi, g))
        if scale != 1.0:
            im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
        tiles.append(im)
    w, h = tiles[0].size
    out = Image.new('L', (w * 2 + 8, h * 2 + 8), 0)
    out.paste(tiles[0], (0, 0))
    out.paste(tiles[1], (w + 8, 0))
    out.paste(tiles[2], (0, h + 8))
    out.paste(tiles[3], (w + 8, h + 8))
    out.save(path)
    print(path, out.size, 'lo %.4e med %.4e sig %.2e 拉伸到 +%.1fsig' % (lo, med, sig, sig_hi))


def main():
    os.makedirs(OUT, exist_ok=True)
    data = [(n, load(p)) for n, p in ITEMS]
    for n, a in data:
        print(n, a.shape)
    h, w = data[0][1].shape
    full = [(n, a, (0, 0, h, w)) for n, a in data]
    grid(full, os.path.join(OUT, 'aligned_sheet.png'), 0.45, 0.5)

    # 1:1 裁切：绕 HFG1 所在的画幅中心偏右下（master 网格）
    cy, cx, half = int(h * 0.37), int(w * 0.40), 400
    sub = (max(0, cy - half), max(0, cx - half), min(h, cy + half), min(w, cx + half))
    grid([(n, a, sub) for n, a in data], os.path.join(OUT, 'aligned_neb.png'), 0.45, 1.0)
    print('crop box', sub)

    # Z2 - F1 差值（两版都已按同一变换重采样到同一网格）
    f1 = dict(data)['F1 bg0'][sub[0]:sub[2], sub[1]:sub[3]]
    z2 = dict(data)['Z2 bg1'][sub[0]:sub[2], sub[1]:sub[3]]
    d = (z2 - f1).astype(np.float64)
    s = float(d.std())
    dd = np.clip(d / (8 * s) + 0.5, 0, 1)
    im = Image.fromarray((dd * 255).astype(np.uint8)).resize((400 * 2, 400 * 2), Image.NEAREST)
    im.save(os.path.join(OUT, 'aligned_diff.png'))
    print('aligned_diff.png  d.std %.3e  d.ptp %.3e' % (s, float(d.max() - d.min())))


if __name__ == '__main__':
    main()
