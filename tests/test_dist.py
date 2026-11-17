# -*- coding: utf-8 -*-
"""径向畸变项自检。

模型：帧坐标 → 参考坐标 = k·R(θ)·warp(p, d2) + u，warp 是 ρ'=ρ(1+d2ρ²)。
所以「帧里带着 d2 的畸变」时，拟合出的系数应当约为 −d2。
"""
import math
import sys

import numpy as np

sys.path.insert(0, 'core')
import drizzle as D


def make(d2_true, n=180, seed=7, h=1200, w=1600, noise=0.1):
    rng = np.random.default_rng(seed)
    c = np.array([w / 2.0, h / 2.0])
    rc = 0.5 * min(h, w)
    r = rc * np.sqrt(rng.uniform(0.02, 1.3, n))
    t = rng.uniform(0, 2 * math.pi, n)
    ref = np.stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)], axis=1)
    frame = D._dist_warp(ref, c, rc, d2_true)
    frame = frame + np.random.default_rng(seed + 1).normal(0, noise, frame.shape)
    return frame, ref, c, rc


def main():
    ok = True
    for d2_true in (0.0, 0.010, -0.014):
        p, q, c, rc = make(d2_true)
        rms_sim = D._fit_sim(p, q)[4]
        th, k, ux, uy, rms, d2_hat = D._fit_sim_dist(p, q, c, rc)
        good = (abs(d2_hat + d2_true) < 0.006 and rms < rms_sim * 0.9) or \
               (d2_true == 0.0 and abs(d2_hat) < 0.004)
        print('真值 d2=%+.3f → 解得 %+.4f  残差 相似 %.3f → 带畸变 %.3f px  %s'
              % (d2_true, d2_hat, rms_sim, rms, 'OK' if good else 'FAIL'))
        ok &= good
    print('dist selftest', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
