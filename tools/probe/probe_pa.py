# -*- coding: utf-8 -*-
"""叠加图星点长轴方向 vs 切向（判断拖影是旋转残差还是别的）。"""
import math
import numpy as np
import xisf_io


def stars(img, thr_pct=99.6):
    from scipy.ndimage import maximum_filter
    mx = maximum_filter(img, 5)
    pts = np.argwhere((img == mx) & (img > np.percentile(img, thr_pct)))
    out = []
    h, w = img.shape
    for y, x in pts:
        if y < 8 or x < 8 or y > h - 9 or x > w - 9:
            continue
        t = img[y - 7:y + 8, x - 7:x + 8].astype(np.float64)
        t = t - np.median(t)
        t[t < 0] = 0
        s = t.sum()
        if s <= 0:
            continue
        yy, xx = np.mgrid[-7:8, -7:8]
        mx_, my_ = (t * xx).sum() / s, (t * yy).sum() / s
        cxx = (t * (xx - mx_) ** 2).sum() / s
        cyy = (t * (yy - my_) ** 2).sum() / s
        cxy = float((t * (xx - mx_) * (yy - my_)).sum() / s)
        cov = np.array([[cxx, cxy], [cxy, cyy]])
        ev, evec = np.linalg.eigh(cov)
        a, b = math.sqrt(max(ev[1], 1e-9)), math.sqrt(max(ev[0], 1e-9))
        if a <= 0:
            continue
        pa = math.degrees(math.atan2(evec[1, 1], evec[0, 1])) % 180.0
        out.append((x, y, a, b, pa))
    return np.array(out)


def main():
    img = np.asarray(xisf_io.read_xisf(r'G:\HFG1\lab\H6_dbg\PWS_H6_dbg.xisf')[0].data, np.float32)
    cy, cx = img.shape[0] / 2.0, img.shape[1] / 2.0
    st = stars(img)
    print('星点', len(st))
    r = np.hypot(st[:, 0] - cx, st[:, 1] - cy)
    # 切向角（与 x 轴夹角，模 180）
    tang = (np.degrees(np.arctan2(st[:, 1] - cy, st[:, 0] - cx)) + 90.0) % 180.0
    d = np.abs(((st[:, 4] - tang) + 90) % 180 - 90)
    for lo, hi in ((0, 400), (400, 900), (900, 1400), (1400, 2000), (2000, 3000)):
        k = (r >= lo) & (r < hi)
        if k.sum() >= 5:
            print('r %4d-%4d n=%3d  轴比 b/a %.3f  长轴偏离切向 中位 %5.1f°  (P25 %5.1f)' % (
                lo, hi, k.sum(), np.median(st[k, 3] / st[k, 2]),
                np.median(d[k]), np.percentile(d[k], 25)))


if __name__ == '__main__':
    main()
