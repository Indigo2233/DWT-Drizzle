# -*- coding: utf-8 -*-
"""单帧 vs 叠加：星点扁率随半径的变化（判断拖影来自数据还是配准）。"""
import sys
import numpy as np
from scipy.spatial import cKDTree
import core.pws as pws
import core.drizzle as dz
import xisf_io


def moments(img, pts, h=6):
    out = []
    for x, y in pts:
        ix, iy = int(round(x)), int(round(y))
        if ix - h < 0 or iy - h < 0 or ix + h + 1 > img.shape[1] or iy + h + 1 > img.shape[0]:
            continue
        t = img[iy - h:iy + h + 1, ix - h:ix + h + 1].astype(np.float64)
        t = t - np.median(t)
        t[t < 0] = 0
        s = t.sum()
        if s <= 0:
            continue
        yy, xx = np.mgrid[-h:h + 1, -h:h + 1]
        mx, my = (t * xx).sum() / s, (t * yy).sum() / s
        cxx = (t * (xx - mx) ** 2).sum() / s
        cyy = (t * (yy - my) ** 2).sum() / s
        cxy = (t * (xx - mx) * (yy - my)).sum() / s
        ev = np.linalg.eigvalsh(np.array([[cxx, cxy], [cxy, cyy]]))
        if ev[1] <= 0:
            continue
        out.append((x, y, np.sqrt(max(ev[0], 1e-9)), np.sqrt(ev[1])))
    return np.array(out)


def profile(tag, img, pts, cy, cx):
    m = moments(img, pts)
    if not len(m):
        print(tag, '无星')
        return
    r = np.hypot(m[:, 0] - cx, m[:, 1] - cy)
    b, a = m[:, 2], m[:, 3]
    print(tag, 'n=%d  整体 FWHM≈%.2f  b/a 中位 %.3f' % (len(m), 2.355 * a.mean(), np.median(b / a)))
    for lo, hi in ((0, 300), (300, 600), (600, 900), (900, 1300), (1300, 1800)):
        k = (r >= lo) & (r < hi)
        if k.sum() >= 4:
            print('   r %4d-%4d n=%3d  a中位 %.2f  b/a %.3f' % (lo, hi, k.sum(), np.median(a[k]), np.median(b[k] / a[k])))


def main():
    files, _ = pws.collect_frames(r'G:\HFG1-2\calibrated')
    groups = pws.group_frames_by_filter(files)
    h = groups['H'][:6]
    fs = pws.prepare_frames(h, 'Hprobe', 0, None)
    cy, cx = np.asarray(fs.shape) / 2.0
    img0 = fs.frame(0)
    pts = dz._detect_stars(img0)
    profile('单帧 #0', img0, pts, cy, cx)
    st = np.asarray(xisf_io.read_xisf(r'G:\HFG1\lab\H6_dbg\PWS_H6_dbg.xisf')[0].data, np.float32)
    # 叠加图是 ×2 网格，把星点坐标乘 2（输出原点与输入的关系由 plan 决定，这里只比扁率）
    pts2 = pts * 2.0
    profile('叠加 H6(×2)', st, pts2, cy * 2, cx * 2)


if __name__ == '__main__':
    main()
