# -*- coding: utf-8 -*-
"""合成星场自检：已知 θ/u 的旋转+平移，走 stack_drizzle，看复原的星点是否发虚。

约定与 core/drizzle.py 一致：out = k·R'(θ)·in + u，R'(θ)=[[ct,-st],[st,ct]]，dx = -u。
"""
import math
import numpy as np
from scipy.ndimage import map_coordinates
import core.drizzle as dz


def synth(shape=(700, 900), seed=3, n=90):
    rng = np.random.default_rng(seed)
    h, w = shape
    img = np.zeros(shape, np.float64) + 100.0
    ys = rng.integers(20, h - 20, n)
    xs = rng.integers(20, w - 20, n)
    yy, xx = np.mgrid[0:h, 0:w]
    for y, x in zip(ys, xs):
        img += 800.0 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.1 ** 2))
    return img


def warp(ref, th_deg, ux, uy):
    """生成一帧：in(p) = ref(R'(-θ)·(p-u))"""
    h, w = ref.shape
    yy, xx = np.mgrid[0:h, 0:w]
    th = math.radians(th_deg)
    ct, st = math.cos(th), math.sin(th)
    # r = R'(θ)·p + u ；in(p) = ref(r)
    rx = ct * xx - st * yy + ux
    ry = st * xx + ct * yy + uy
    return map_coordinates(ref, [ry, rx], order=3, mode='constant', cval=100.0)


class Stub:
    def __init__(self, arr):
        self.arr = arr.astype(np.float32)
        self.shape = self.arr.shape[1:]
        self.off = np.zeros(len(arr))
        self.bgcoef = None
        self.bgdeg = 0
    def __len__(self):
        return len(self.arr)
    def strip(self, i, y0, y1):
        return self.arr[i, y0:y1]


def fwhm_map(img):
    """用二阶矩估计中心星点的平均 FWHM 与扁率。"""
    from scipy.ndimage import maximum_filter
    mx = maximum_filter(img, 7)
    pts = np.argwhere((img == mx) & (img > np.percentile(img, 99.5)))
    a_all, e_all = [], []
    for y, x in pts:
        if y < 8 or x < 8 or y > img.shape[0] - 9 or x > img.shape[1] - 9:
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
        cxy = (t * (xx - mx_) * (yy - my_)).sum() / s
        ev = np.linalg.eigvalsh(np.array([[cxx, cxy], [cxy, cyy]]))
        a_all.append(2.355 * math.sqrt(max(ev[1], 1e-9)))
        e_all.append(math.sqrt(max(ev[0], 1e-9) / max(ev[1], 1e-9)))
    return (float(np.median(a_all)), float(np.median(e_all)), len(a_all)) if a_all else (np.nan, np.nan, 0)


def main():
    ref = synth()
    ths = [0.0, 0.03, -0.02, 0.05, -0.04, 0.01]
    uxs = [0.0, 2.3, -1.1, 0.7, -2.0, 1.3]
    uys = [0.0, -1.4, 2.2, -0.6, 1.8, -2.4]
    frames = [warp(ref, t, ux, uy) for t, ux, uy in zip(ths, uxs, uys)]
    print('输入单帧 FWHM/e/n =', fwhm_map(frames[0]))
    dx = np.array([-u for u in uxs], float)
    dy = np.array([-u for u in uys], float)
    theta = np.array(ths, float)
    plan = dz.plan_output(ref.shape, dx, dy, 1, margin=6.0, theta=theta)
    R = np.ones((plan['H'], plan['W']), np.float64)
    C = np.ones(len(frames), np.float32)
    S = np.ones(len(frames), np.float32)
    out = dz.stack_drizzle(Stub(np.stack(frames)), R, C, S, dx, dy, plan,
                           A=20.0, B=2.0, rej_k=3.5, rej_gate=0.0, sky=100.0,
                           pixfrac=1.0, band_h=120, theta=theta, scale=np.ones(len(frames)),
                           native_shape=ref.shape, workers=0)['stack']
    print('输出 FWHM/e/n =', fwhm_map(out))
    # 对照：只做平移、不旋转（θ 传 0）
    out2 = dz.stack_drizzle(Stub(np.stack(frames)), R, C, S, dx, dy, plan,
                            A=20.0, B=2.0, rej_k=3.5, rej_gate=0.0, sky=100.0,
                            pixfrac=1.0, band_h=120, theta=np.zeros(len(frames)),
                            scale=np.ones(len(frames)),
                            native_shape=ref.shape, workers=0)['stack']
    print('纯平移 FWHM/e/n =', fwhm_map(out2))


if __name__ == '__main__':
    main()
