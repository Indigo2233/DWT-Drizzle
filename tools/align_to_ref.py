"""把 drizzle 成品对齐并裁切到参考帧（master）的网格上。

用法：
  python tools/align_to_ref.py --src G:\\HFG1\\drz_o4\\PWS_o4.xisf \\
      --ref "G:\\HFG1-2\\master\\masterLight_..._autocrop.xisf" \\
      --out G:\\HFG1\\drz_o4 --tag o4

思路：src 是 ×2 输出（6115×5128），先 2×2 块平均回原生尺度，与 ref 做刚体
配准（θ, u），再把 **×2 原图** 用一次 affine_transform 重采样到 ref 的网格，
所以只插值一次，没有二次损失。输出两个版本：原生网格与 ×2 网格。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
_C1 = HERE.parent / 'upstream' / 'core'
_C2 = HERE.parent / 'core'
CORE = _C1 if _C1.exists() else _C2
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(CORE.parent))
sys.path.insert(0, str(HERE.parent / 'upstream'))


def block_mean(a: np.ndarray, s: int) -> np.ndarray:
    if s <= 1:
        return a
    h, w = (a.shape[0] // s) * s, (a.shape[1] // s) * s
    return a[:h, :w].reshape(h // s, s, w // s, s).mean(axis=(1, 3))


def _match(rot: np.ndarray, ref: np.ndarray, u, tree, rad):
    q = rot + np.asarray(u)
    d, idx = tree.query(q, distance_upper_bound=rad)
    ok = np.isfinite(d)
    return ok, idx, d


def fit_rigid(ref_img: np.ndarray, src_img: np.ndarray, log=print):
    """ref = R(θ)·src + u，返回 (θ_deg, u=(uy,ux), rms, n)。"""
    from drizzle import _detect_stars, _rot_hist, _fit_rigid, rot_rows_np

    s_ref = _detect_stars(ref_img, n=260)
    s_src = _detect_stars(src_img, n=260)
    log(f'星点：参考 {len(s_ref)}，待对齐 {len(s_src)}')
    if len(s_ref) < 5 or len(s_src) < 5:
        raise SystemExit('星点太少，无法配准')

    tree = cKDTree(s_ref)
    h = _rot_hist(s_ref, s_src)
    if h is None:
        raise SystemExit('星点对角度谱失败：两张图可能不是同一天区')
    th = float(h[0])
    log(f'角度谱：θ={th:+.3f}°（{h[1]:.0f} 票）')

    # 平移用"两两位移向量取众数"求，比先转再最近邻稳
    rot = rot_rows_np(s_src, th)
    D = (s_ref[None, :, :] - rot[:, None, :]).reshape(-1, 2)
    qb = np.round(D / 2.0).astype(np.int64)
    key = qb[:, 0] * 100000 + qb[:, 1]
    uniq, cnt = np.unique(key, return_counts=True)
    k = uniq[int(np.argmax(cnt))]
    u = np.array([k // 100000, k % 100000], float) * 2.0   # (dx, dy)
    log(f'位移众数：u=(dx {u[0]:+.1f}, dy {u[1]:+.1f})px（{cnt.max()} 票 / {len(D)}）')
    for rad in (8.0, 5.0, 3.0):
        ok, idx, d = _match(rot, s_ref, u, tree, rad)
        if ok.sum() < 8:
            raise SystemExit('刚体配准失败（匹配星点太少）')
        u = np.median(s_ref[idx[ok]] - rot[ok], axis=0)
    log(f'  初配：{int(ok.sum())} 对，中位残差 {np.median(d[ok]):.3f}px')

    for _ in range(4):
        rot = rot_rows_np(s_src, th)
        ok, idx, d = _match(rot, s_ref, u, tree, 3.0)
        if ok.sum() < 8:
            break
        p = rot[ok]
        q = s_ref[idx[ok]]
        u = np.median(q - p, axis=0)
        th_n, ux, uy, rms = _fit_rigid(p, q)
        if not np.isfinite(th_n):
            break
        th = th + float(th_n)
        if abs(th_n) < 1e-5 and rms < 0.20:
            break
    rot = rot_rows_np(s_src, th)
    ok, idx, d = _match(rot, s_ref, u, tree, 3.0)
    if ok.sum() >= 8:
        u = np.median(s_ref[idx[ok]] - rot[ok], axis=0)
    rms = float(np.sqrt(np.mean(d[ok] ** 2))) if ok.sum() else float('nan')
    log(f'刚体解：θ={th:+.3f}°  u=(dx {u[0]:+.3f}, dy {u[1]:+.3f})px  '
        f'残差 rms {rms:.3f}px  匹配 {int(ok.sum())}/{len(s_src)}')
    return th, u, rms, int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='aligned')
    ap.add_argument('--scale', type=float, default=2.0)
    ap.add_argument('--no-2x', action='store_true')
    a = ap.parse_args()

    from xisf_io import read_xisf, write_xisf

    Path(a.out).mkdir(parents=True, exist_ok=True)
    logf = open(Path(a.out) / f'{a.tag}_align.log', 'w', encoding='utf-8')

    def log(*m):
        s = ' '.join(str(x) for x in m)
        print(s)
        logf.write(s + '\n')
        logf.flush()

    ref_i = read_xisf(a.ref)[0]
    src_i = read_xisf(a.src)[0]
    ref = np.asarray(ref_i.data, np.float64)
    src = np.asarray(src_i.data, np.float64)
    log(f'参考 {ref.shape}  待对齐 {src.shape}  ×{a.scale:g}')

    if abs(a.scale - round(a.scale)) < 1e-6:
        down = block_mean(src, int(round(a.scale)))
    else:
        # 非整数倍（如 1.5×）：用一次插值降采样到参考网格，只用于测配准
        down = ndimage.zoom(src, 1.0 / float(a.scale), order=1)
    th, u, rms, nmatch = fit_rigid(ref, down, log=log)

    c, sn = np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th))
    # (x, y) 序：q = R(θ)p + u  ⇒  p = R(−θ)(q − u)
    R_inv = np.array([[c, sn], [-sn, c]], float)

    def render(scale_out: float) -> np.ndarray:
        # 输出网格 q（matster 像素，(y,x) 数组序）→ 源图坐标
        Mxy = (a.scale / scale_out) * R_inv
        off_xy = -a.scale * (R_inv @ u)
        M = Mxy[::-1, ::-1]          # (x,y) 矩阵 → (row,col) 序
        off = off_xy[::-1]
        shp = (int(round(ref.shape[0] * scale_out)), int(round(ref.shape[1] * scale_out)))
        order = 3      # 三次样条：2:1 下采样和亚像素旋转都够平滑
        out = ndimage.affine_transform(src, M, offset=off, output_shape=shp,
                                       order=order, mode='constant', cval=0.0)
        return out.astype(np.float32)

    out1 = render(1.0)
    # 用星点精配准把残差再吃掉一轮（刚体解只有 ~0.4px 精度）
    for it in range(2):
        try:
            th2, u2, rms2, n2 = fit_rigid(ref, out1, log=(lambda *m: None))
        except SystemExit:
            break
        log(f'残差配准 #{it}: θ={th2:+.4f}° u=({u2[0]:+.3f},{u2[1]:+.3f}) rms={rms2:.3f}px n={n2}')
        if abs(th2) < 1e-4 and np.hypot(*u2) < 0.05:
            break
        c2, s2 = np.cos(np.deg2rad(th2)), np.sin(np.deg2rad(th2))
        R2 = np.array([[c2, -s2], [s2, c2]])
        th = th + th2
        u = R2 @ u + u2
        out1 = render(1.0)
    cov = float((out1 > 0).mean())
    log(f'原生网格输出 {out1.shape}，非零覆盖 {cov:.2%}')

    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)

    img = ref_i
    img.data = np.ascontiguousarray(out1, dtype=np.float32)
    img.geometry = (out1.shape[1], out1.shape[0], 1)
    img.sample_format = 'Float32'
    img.color_space = 'Gray'
    p1 = o / f'{a.tag}_aligned.xisf'
    write_xisf(p1, img)
    log(f'写出 {p1}  {p1.stat().st_size} B')

    if not a.no_2x:
        out2 = render(2.0)
        img2 = ref_i
        img2.data = np.ascontiguousarray(out2, dtype=np.float32)
        img2.geometry = (out2.shape[1], out2.shape[0], 1)
        img2.sample_format = 'Float32'
        img2.color_space = 'Gray'
        p2 = o / f'{a.tag}_aligned_2x.xisf'
        write_xisf(p2, img2)
        log(f'写出 {p2}  {p2.stat().st_size} B')

    log(f'完成：θ={th:+.3f}°  u=(dx {u[0]:+.3f}, dy {u[1]:+.3f})  残差 {rms:.3f}px')
    logf.close()


if __name__ == '__main__':
    main()
