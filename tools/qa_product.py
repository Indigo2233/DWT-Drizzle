"""成品体检：星点 FWHM、背景噪声、覆盖、与 master 的对比图。

用法：
  python tools/qa_product.py --img G:\\HFG1\\drz_o4\\o4_aligned.xisf \
      --ref "...masterLight...xisf" --out G:\\HFG1\\drz_o4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'upstream'))
sys.path.insert(0, str(HERE.parent / 'upstream' / 'core'))


def radial_fwhm(img, x, y, half=25):
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    r = np.hypot(yy, xx)
    c = img[y - half:y + half + 1, x - half:x + half + 1].astype(np.float64)
    if c.shape != r.shape or y - half < 0 or x - half < 0:
        return None
    bkg = np.median(c[r > half * 0.8])
    prof = np.array([c[(r >= k) & (r < k + 1)].mean() for k in range(half)])
    prof = prof - bkg
    if prof[0] <= 0:
        return None
    pk = prof.max()
    above = np.nonzero(prof >= 0.5 * pk)[0]
    if not len(above):
        return None
    return 2.0 * (np.interp(0.5 * pk, prof[::-1], np.arange(len(prof))[::-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img', required=True)
    ap.add_argument('--ref', default='')
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    from xisf_io import read_xisf
    from drizzle import _detect_stars

    img = np.asarray(read_xisf(a.img)[0].data, np.float64)
    print(f'图像 {img.shape}  范围 {img.min():.4g}~{img.max():.4g}  非零 {(img > 0).mean():.2%}')

    stars = _detect_stars(img, n=400)
    vals = []
    for x, y in stars:
        x, y = int(round(x)), int(round(y))
        if 30 < x < img.shape[1] - 30 and 30 < y < img.shape[0] - 30:
            f = radial_fwhm(img, x, y)
            if f and 1.0 < f < 20.0:
                vals.append(f)
    vals = np.array(vals)
    if len(vals):
        print(f'星点 {len(vals)} 颗  FWHM 中位 {np.median(vals):.2f}px  '
              f'四分位 {np.percentile(vals, 25):.2f}~{np.percentile(vals, 75):.2f}px')

    if a.ref:
        ref = np.asarray(read_xisf(a.ref)[0].data, np.float64)
        print(f'参考 {ref.shape}')
        rs = _detect_stars(ref, n=400)
        rv = []
        for x, y in rs:
            x, y = int(round(x)), int(round(y))
            if 30 < x < ref.shape[1] - 30 and 30 < y < ref.shape[0] - 30:
                f = radial_fwhm(ref, x, y)
                if f and 1.0 < f < 20.0:
                    rv.append(f)
        rv = np.array(rv)
        if len(rv):
            print(f'参考星点 {len(rv)} 颗  FWHM 中位 {np.median(rv):.2f}px')

        m = min(img.shape[0], ref.shape[0])
        sky = slice(m // 2 - 150, m // 2 + 150)
        def mad(x):
            return 1.4826 * np.median(np.abs(x - np.median(x)))
        print(f'背景噪声  成品 {mad(img[sky, sky]):.4e}  参考 {mad(ref[sky, sky]):.4e}')

        if a.out:
            from PIL import Image
            def norm(x):
                lo, hi = np.percentile(x[x > 0], [25, 99.9])
                return (np.clip((x - lo) / (hi - lo), 0, 1) ** 0.45 * 255).astype(np.uint8)
            h, w = img.shape
            cy, cx = h // 2, w // 2
            s = 220
            A = norm(ref[cy - s:cy + s, cx - s:cx + s])
            B = norm(img[cy - s:cy + s, cx - s:cx + s])
            im = Image.new('L', (s * 4 + 10, s * 2))
            im.paste(Image.fromarray(A), (0, 0))
            im.paste(Image.fromarray(B), (s * 2 + 10, 0))
            p = Path(a.out) / 'qa_compare.png'
            im.save(p)
            print(f'对比图 {p}')


if __name__ == '__main__':
    main()
