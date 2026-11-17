# -*- coding: utf-8 -*-
"""把一个 ×2 的 drizzle 成品对齐/裁切到某张参考图的网格上（参考图本身也是 ×2）。

与 tools/align_to_ref.py 的区别：先用 FFT 相位相关拿粗平移，再用星点做刚体精配准，
所以不依赖"角度谱 + 位移直方图"那套初始搜索（两图背景电平差很大时它会失败）。
"""
import sys

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

sys.path.insert(0, 'core')
import xisf_io
import probe_pa as P


def load(p):
    return np.asarray(xisf_io.read_xisf(p)[0].data, np.float64)


def block_mean(a, s=2):
    h, w = (a.shape[0] // s) * s, (a.shape[1] // s) * s
    return a[:h, :w].reshape(h // s, s, w // s, s).mean(axis=(1, 3))


def coarse_shift(a, b, ds=8):
    """返回 (dy, dx)：b 相对 a 的平移（b[y+dy, x+dx] ≈ a[y, x]）。"""
    A = ndimage.gaussian_filter(np.maximum(a[::ds, ::ds], 0), 1.0)
    B = ndimage.gaussian_filter(np.maximum(b[::ds, ::ds], 0), 1.0)
    A = A - A.mean()
    B = B - B.mean()
    c = fftconvolve(A, B[::-1, ::-1], mode='full')
    py, px = np.unravel_index(np.argmax(c), c.shape)
    return (py - (B.shape[0] - 1)) * ds, (px - (B.shape[1] - 1)) * ds


def fit_rigid(p, q):
    pc, qc = p - p.mean(0), q - q.mean(0)
    H = pc.T @ qc
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(2)
    if np.linalg.det(Vt.T @ U.T) < 0:
        D[1, 1] = -1.0
    R = Vt.T @ D @ U.T
    u = q.mean(0) - R @ p.mean(0)
    th = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    r = q - (p @ R.T + u)
    return th, u, float(np.sqrt(np.mean(np.sum(r * r, 1)))), R


def main():
    src_p = sys.argv[1]      # 待对齐（×2）
    ref_p = sys.argv[2]      # 参考（×2，决定输出网格）
    out_p = sys.argv[3]      # 输出 xisf
    tag = sys.argv[4] if len(sys.argv) > 4 else 'aligned'

    ref = load(ref_p)
    src = load(src_p)
    print('参考 %s  源 %s' % (ref.shape, src.shape))
    ref_n = block_mean(ref, 2)
    src_n = block_mean(src, 2)
    dy, dx = coarse_shift(ref_n, src_n)
    print('FFT 粗配准：源在参考里的位移 dy=%d dx=%d（原生像素）' % (dy, dx))

    sr = P.stars(ref_n.astype(np.float32))
    ss = P.stars(src_n.astype(np.float32))
    print('星点 参考 %d 源 %d' % (len(sr), len(ss)))
    # 源星图 → 参考坐标（原生）
    pred = ss[:, :2] + [dx, dy]
    tree = cKDTree(sr[:, :2])
    best = None
    for rad in (3.0, 5.0, 8.0):
        d, idx = tree.query(pred, k=1, distance_upper_bound=rad)
        ok = np.isfinite(d)
        if ok.sum() >= 12 and (best is None or ok.sum() > best[0]):
            best = (int(ok.sum()), ss[ok, :2], sr[idx[ok], :2])
    if best is None or best[0] < 12:
        sys.exit('星点匹配不足，无法精配准')
    n, p, q = best
    th, u, rms, R = fit_rigid(p, q)
    print('刚体解 θ=%+.4f° u=(%+.2f, %+.2f) 残差 %.3fpx 匹配 %d 对' % (th, u[0], u[1], rms, n))
    # 迭代一次（用新解重新找最近邻）
    pred = p @ R.T + u
    d, idx = cKDTree(sr[:, :2]).query(pred, k=1, distance_upper_bound=2.0)
    ok = np.isfinite(d)
    if ok.sum() >= 12:
        th, u, rms, R = fit_rigid(p[ok], sr[idx[ok], :2])
        print('精修 θ=%+.4f° u=(%+.2f, %+.2f) 残差 %.3fpx 匹配 %d 对'
              % (th, u[0], u[1], rms, int(ok.sum())))

    c, s = np.cos(np.radians(th)), np.sin(np.radians(th))
    # q = R·p + u（原生坐标）⇒ p = R⁻¹(q − u)，再 ×2 到源图坐标
    # 输出索引是参考的 ×2 像素，而 u 是原生像素：q2x/2 − u 再 ×2 ⇒ 旋转矩阵不乘 2，
    # 只有平移项乘 2。
    M = np.array([[c, -s], [s, c]])
    off = np.array([-2 * (-s * u[0] + c * u[1]), -2 * (c * u[0] + s * u[1])])
    out = ndimage.affine_transform(src, M, offset=off, output_shape=ref.shape,
                                   order=3, mode='constant', cval=0.0).astype(np.float32)
    cov = float((out > 0).mean())
    print('输出 %s（参考网格）非零覆盖 %.2f%%' % (str(ref.shape), cov * 100))

    img = xisf_io.read_xisf(ref_p)[0]
    img.data = np.ascontiguousarray(out)
    img.geometry = (out.shape[1], out.shape[0], 1)
    img.sample_format = 'Float32'
    img.color_space = 'Gray'
    xisf_io.write_xisf(out_p, img)
    print('写出', out_p)

    # 对齐后再测一次星点残差（用重采样后的图 vs 参考）
    o = P.stars(out.astype(np.float32))
    pred = o[:, :2]
    d, idx = cKDTree(sr[:, :2]).query(pred, k=1, distance_upper_bound=3.0)
    ok = np.isfinite(d)
    if ok.sum() >= 12:
        _, _, rms2, _ = fit_rigid(o[ok, :2], sr[idx[ok], :2])
        print('重采样后残余 rms %.3fpx（%d 对）' % (rms2, int(ok.sum())))


if __name__ == '__main__':
    main()
