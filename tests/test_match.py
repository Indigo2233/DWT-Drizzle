# -*- coding: utf-8 -*-
"""matchref 自检：造一张参考图，再按已知的「旋转+平移+2 倍像素」造成品，看能不能解回来。

    python tests/test_match.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'core'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matchref


def star_field(h=400, w=420, n=140, seed=7):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), np.float64) + 0.002
    ys = rng.integers(12, h - 12, n)
    xs = rng.integers(12, w - 12, n)
    amps = rng.uniform(0.05, 3.0, n)
    for y, x, a in zip(ys, xs, amps):
        img[y, x] += a
    return ndimage.gaussian_filter(img, 1.2)


def main():
    ref = star_field()
    th_true, sh_true = 1.5, (12.0, -7.0)      # 成品相对参考：旋转 1.5°、平移 (dx, dy)
    c, s = np.cos(np.radians(th_true)), np.sin(np.radians(th_true))
    R = np.array([[c, -s], [s, c]])
    # 正向：参考坐标 q 映射到成品（原生）坐标 p = R·q + u
    M = R.T
    off = -R.T @ np.array(sh_true)
    native = ndimage.affine_transform(ref, M, offset=off, output_shape=(420, 440),
                                      order=1, mode='constant', cval=0.0)
    src = np.repeat(np.repeat(native, 2, 0), 2, 1)      # 成品是 ×2 像素

    out, info = matchref.align(src, ref, log=print)
    assert out is not None, info
    print('解出 k=%d θ=%+.3f° 位移=(%+.2f, %+.2f) 残差 %.3fpx 匹配 %d 对 覆盖 %.1f%%'
          % (info['k'], info['theta'], info['u'][0], info['u'][1], info['rms'],
             info['n_pair'], info['cover'] * 100))
    assert abs(info['k'] - 2) < 0.05, '应识别出成品是 2 倍像素（得到 %.3f）' % info['k']
    assert abs(abs(info['theta']) - th_true) < 0.25, '旋转角解错了'
    assert info['rms'] < 1.0, '星点残差太大'
    assert info['cover'] > 0.95, '覆盖太少'

    # 拿回参考网格后，星点应该落在参考星点附近
    sr = matchref.find_stars(ref)
    so = matchref.find_stars(out.astype(np.float64))
    from scipy.spatial import cKDTree
    d, _ = cKDTree(sr).query(so, k=1)
    d = np.sort(d)[:min(len(d), len(sr))]
    print('重采样后星点偏差 中位 %.3fpx  P90 %.3fpx' % (np.median(d), np.percentile(d, 90)))
    assert np.median(d) < 0.6, '重采样后星点没落回参考位置'

    m = (out > 0) & (ref > 0)
    corr = float(np.corrcoef(out[m], ref[m])[0, 1])
    print('与参考图相关系数 %.4f' % corr)
    assert corr > 0.92, '重采样后的图与参考不像（几何没对上）'
    print('matchref selftest 通过')


if __name__ == '__main__':
    main()
