"""CFA Drizzle（Bayer drizzle）：不解马赛克，直接把原始 CFA 像素按颜色分平面叠加。

设计要点（每条都是硬约束，改了就失去意义）：
  · 不 debayer：R 像素只进 R 平面、G 只进 G、B 只进 B。配准只是在 G 平面上
    "量几何"，量出来的 dx/dy/θ/k 三平面共用。
  · drop 尺寸 P = pixfrac·s·k，pixfrac 固定 1.0 → 2× 时一个输入像素正好铺满
    2×2 个输出像素，不重叠也不留缝（1× 会留棋盘格，4× 抖动撑不满）。
  · 逐平面排异：R/B 每像素只有 n/4 个样本、G 有 n/2，阈值必须各算各的。
  · PWS 的语义不变：C（清晰度）、S（信噪比）、R 场（空间权重）照旧，
    只是从"一张灰度图"变成"三张稀疏平面"。

几何全部复用 core/drizzle.py 的 plan_output_rot / _xy_rows / _scatter_band，
与正式叠加同一份代码——不要在这里另写坐标公式。
"""
from __future__ import annotations

import os
import time
from pathlib import Path
import warnings
from typing import Dict, List, Optional

import numpy as np


def _colmap(pattern: str) -> np.ndarray:
    m = np.zeros((2, 2), np.int64)
    for yy in range(2):
        for xx in range(2):
            m[yy, xx] = 'RGB'.index(pattern[yy * 2 + xx])
    return m


# ---------------------------------------------------------------------------
# 分带并行：每个进程处理一条输出带，只读 memmap，返回本带的三平面累加
# ---------------------------------------------------------------------------
_G = {}
_PF = np.zeros((0, 4), np.float64)


def _splat_quad(vals, X, Y, W, Y0, Y1, sk, theta, n=2):
    """旋转正方形内 n×n 点采样，最近像素落点累加（面积覆盖近似）。"""
    bh = Y1 - Y0
    size = bh * W
    v = np.asarray(vals, np.float64)
    cs, sn = float(np.cos(theta)), float(np.sin(theta))
    offs = (np.arange(n, dtype=np.float64) + 0.5) / n - 0.5
    wt = 1.0 / (n * n)
    acc = np.zeros(size, np.float64)
    cov = np.zeros(size, np.float64)
    for oy in offs:
        for ox in offs:
            Xs = X + (cs * ox - sn * oy) * sk
            Ys = Y + (sn * ox + cs * oy) * sk
            ix = np.floor(Xs).astype(np.int64)
            iy = np.floor(Ys).astype(np.int64)
            ok = (ix >= 0) & (ix < W) & (iy >= Y0) & (iy < Y1)
            if not ok.any():
                continue
            idx = (iy[ok] - Y0) * W + ix[ok]
            acc += np.bincount(idx, weights=v[ok] * wt, minlength=size)
            cov += np.bincount(idx, weights=np.full(idx.size, wt), minlength=size)
    return (acc.reshape(bh, W).astype(np.float32),
            cov.reshape(bh, W).astype(np.float32))


def _splat_tri(vals, X, Y, W, Y0, Y1, sup=2.0):
    """三角核散射：每个样本按「同色栅格间隔」的宽度铺开，和恒为常数。

    方框核（_scatter_band）在同色样本间隔 4 输出像素时会留下周期性的行/列
    权重起伏，实测表现为均匀分布的横向条纹；三角形核的重叠和是常数，
    行方向的权重不再随相位跳变。
    """
    bh = Y1 - Y0
    size = bh * W
    sup = max(0.5, float(sup))     # 三角核半宽（输出像素）
    kx0 = np.floor(X - sup).astype(np.int64)
    ky0 = np.floor(Y - sup).astype(np.int64)
    idxs, wgts, wvts = [], [], []
    v = np.asarray(vals, np.float64)
    for a in range(5):
        kx = kx0 + a
        wx = 1.0 - np.abs((kx + 0.5) - X) / (sup + 0.5)
        np.clip(wx, 0.0, None, out=wx)
        vx = (kx >= 0) & (kx < W) & (wx > 0)
        if not vx.any():
            continue
        for b in range(5):
            ky = ky0 + b
            wy = 1.0 - np.abs((ky + 0.5) - Y) / (sup + 0.5)
            np.clip(wy, 0.0, None, out=wy)
            mm = vx & (ky >= Y0) & (ky < Y1) & (wy > 0)
            if not mm.any():
                continue
            w = wx[mm] * wy[mm]
            idxs.append((ky[mm] - Y0) * W + kx[mm])
            wgts.append(w)
            wvts.append(w * v[mm])
    if not idxs:
        z = np.zeros((bh, W), np.float32)
        return z, z.copy()
    idx = np.concatenate(idxs)
    acc = np.bincount(idx, weights=np.concatenate(wvts), minlength=size).reshape(bh, W)
    cov = np.bincount(idx, weights=np.concatenate(wgts), minlength=size).reshape(bh, W)
    return acc.astype(np.float32), cov.astype(np.float32)


def _band_init(g):
    global _PF
    _G.clear()
    _G.update(g)
    _PF = np.asarray(g.get('pfac', np.zeros((0, 4))), np.float64)


def _band_one(band):
    y0, y1 = int(band[0]), int(band[1])
    import cfa  # noqa: F401  （保持与主进程一致的导入路径）
    from drizzle import _xy_rows
    import pws

    g = _G
    n, h, w, W, H = g['n'], g['h'], g['w'], g['W'], g['H']
    s, pixfrac = g['s'], g['pixfrac']
    dx, dy, th, kk, ok = g['dx'], g['dy'], g['th'], g['kk'], g['ok']
    C, S, A, B, rej_k = g['C'], g['S'], g['A'], g['B'], g['rej_k']
    R_out, plan, colmap, allrows = g['R_out'], g['plan'], g['colmap'], g['allrows']
    mm = np.memmap(g['mmpath'], dtype=np.float32, mode='r', shape=(n, h, w))
    off = np.asarray(g.get('off', np.zeros(n)), np.float64)
    phase = int(os.environ.get('DWT_CFA_PHASE', g.get('phase', 1)) or 0)
    sub = int(os.environ.get('DWT_CFA_SUB', 0) or g.get('sub', 1))
    pad = int(os.environ.get('DWT_CFA_PAD', 0) or (np.ceil(max(1.0, 2.0 * max(
        (g['pixfrac'] * g['s']) * (kk[i] if np.isfinite(kk[i]) else 1.0) for i in range(n)))) + 2))

    bh = y1 - y0
    y0p, y1p = y0 - pad, y1 + pad
    ph = bh + 2 * pad
    if phase:
        acc = np.zeros((3, 4, ph, W), np.float32)
        cnt = np.zeros((3, 4, ph, W), np.float32)
    else:
        num = np.zeros((3, n, ph, W), np.float32)
        wts = np.zeros((3, n, ph, W), np.float32)

    # 逐帧 PWS 权重（含 R 场）；相位模式下按桶累加，权重只影响组内相对权重
    # 只对本带算权重：整幅算会分配 (n, H, W) float64（60 帧 ≈ 5 GiB），多进程直接 OOM。
    # R 场来自解拜耳后的参考栈，双线性插值会在 2 像素尺度留下棋盘残留；
    # 直接拿它做指数权重会把那层残留放大进成品 → 先做一次小幅低通把它洗掉。
    _Rs = np.asarray(R_out[y0:y1], np.float64)
    _rb = float(os.environ.get('DWT_CFA_RBLUR', 1.2) or 0.0)
    if _rb > 0:
        import scipy.ndimage as _ndi
        _Rs = _ndi.gaussian_filter(_Rs, _rb, mode='nearest')
    Wb_band = pws.strip_weights(C, S, _Rs, A, B).astype(np.float32)
    bucket = np.zeros(n, np.int64)
    if phase:
        for i in range(n):
            fx = float(s * dx[i] % 1.0)
            fy = float(s * dy[i] % 1.0)
            bucket[i] = (2 if fx > 0.5 else 0) + (1 if fy > 0.5 else 0)

    used = np.zeros(n, bool)
    for i in range(n):
        if not ok[i]:
            continue
        k_i = float(kk[i]) if np.isfinite(kk[i]) else 1.0
        P = pixfrac * s * k_i * 2.0
        ux, uy = -float(dx[i]), -float(dy[i])
        _sy = s * k_i
        _sh = abs(np.sin(float(th[i]))) * w + 4.0
        j0 = max(0, int(np.floor((y0p - P) / _sy - uy)) - int(_sh) - 2)
        j1 = min(h, int(np.ceil((y1p + P) / _sy - uy)) + int(_sh) + 2)
        if j1 <= j0:
            continue
        X, Y = _xy_rows(j0, j1, w, plan, float(th[i]), ux, uy, k_i, 0.0, h)
        keep = (Y > y0p - P) & (Y < y1p + P)
        yy0 = np.arange(j0, j1)[:, None] % 2
        colr0 = colmap[yy0, allrows[:, :w]]
        raw = np.array(mm[i, j0:j1, :], dtype=np.float32, copy=True)   # 可写副本
        if _PF.shape[0] > i and not np.allclose(_PF[i], 1.0):
            raw *= _PF[i][colr0].astype(np.float32)   # 相位中位对齐
        vals = raw - np.float32(off[i])
        if g.get('bgdeg', 0) > 0 and g.get('bgcoef') is not None:
            import bggrad
            _rc = None
            if g.get('bgrdeg', 0) > 0 and g.get('bgrcoef') is not None:
                _rc = g['bgrcoef'][i]
            vals -= bggrad.model_strip(g['bgcoef'][i], g['bgdeg'], j0, j1, w,
                                       g['bgshape'][0], rcoef=_rc)
        colr = colr0
        used[i] = True
        _cs, _sn = np.cos(float(th[i])), np.sin(float(th[i]))
        _sk = s * k_i
        _o = (np.arange(sub) + 0.5) / sub - 0.5 if sub > 1 else (0.0,)
        if phase:
            wp = np.ones((ph, W), np.float32)
            wp[pad:pad + bh] = Wb_band[i]
        for c in range(3):
            m = (colr == c) & keep
            if not m.any():
                continue
            # 实验：检查 X/Y 是像素角点还是中心点（差半像素会让 2 像素栅格交替错位）
            _hs = float(os.environ.get('DWT_CFA_HALF', '0') or 0)
            if _hs:
                X = X + _hs
                Y = Y + _hs
            nb, cb = _splat_quad(vals[m], X[m], Y[m], W, y0p, y1p,
                                   _sk, float(th[i]),
                                   n=int(os.environ.get('DWT_CFA_QUAD', 0) or 2))
            if phase:
                b = int(bucket[i])
                acc[c, b] += nb * wp
                cnt[c, b] += cb * wp
            else:
                num[c, i] = nb
                wts[c, i] = cb

    sl = slice(pad, pad + bh)
    out = np.zeros((3, bh, W), np.float32)
    cov = np.zeros((3, bh, W), np.float32)
    rej_hit = rej_tot = 0.0
    if phase:
        for c in range(3):
            vs, ms = [], []
            for b in range(4):
                cb_ = cnt[c, b][sl]
                nb_ = acc[c, b][sl]
                m_ = cb_ > 1e-9
                vs.append(np.where(m_, nb_ / np.maximum(cb_, 1e-9), 0.0))
                ms.append(m_.astype(np.float32))
            V = np.dstack(vs)
            M = np.dstack(ms)
            nvb = M.sum(axis=2)
            with np.errstate(all='ignore'):
                val = (V * M).sum(axis=2) / np.maximum(nvb, 1e-9)
            out[c] = np.where(nvb > 0.5, val, np.nan).astype(np.float32)
            cov[c] = np.nansum(cnt[c], axis=0)[sl]
        return y0, out, cov, rej_hit, rej_tot

    # ---- 兼容旧路径（DWT_CFA_PHASE=0）----
    Wb = pws.strip_weights(C, S, np.asarray(R_out[y0:y1], np.float64),
                           A, B).astype(np.float32)
    for c in range(3):
        wgt = wts[c][:, sl, :] * Wb
        v = np.where(wgt > 1e-9, num[c][:, sl, :] / np.maximum(wgt, 1e-9), np.nan)
        if rej_k > 0 and int(used.sum()) >= 5:
            import warnings
            with np.errstate(all='ignore'), warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                anchor = np.nanmedian(v, axis=0)
                dev = np.abs(v - anchor[None])
                _nv = np.isfinite(v).sum(axis=0).astype(np.float64)
                _blk = 16
                _bh2, _bw2 = dev.shape[1:]
                _ph2, _pw2 = (-_bh2) % _blk, (-_bw2) % _blk
                _sp = np.pad(np.nanmedian(dev, axis=0), ((0, _ph2), (0, _pw2)), mode='edge')
                _sp = _sp.reshape(_sp.shape[0] // _blk, _blk, _sp.shape[1] // _blk, _blk)
                _sp = np.nanmedian(_sp, axis=(1, 3))
                sc = np.repeat(np.repeat(_sp, _blk, axis=0), _blk, axis=1)[:_bh2, :_bw2]
                _k_eff = rej_k * np.sqrt(1.0 + 6.0 / np.maximum(_nv, 1.0))
                _k_eff = np.where(_nv >= 6.0, _k_eff, np.inf)
                bad = dev > (_k_eff[None] * sc[None])
            rej_hit += float(np.nansum(bad))
            rej_tot += float(np.sum(_nv))
            wgt = np.where(bad, 0.0, wgt).astype(np.float32)
        out[c] = np.nansum(np.nan_to_num(v) * wgt, axis=0)
        cov[c] = wgt.sum(axis=0)
    return y0, out, cov, rej_hit, rej_tot


def run_group(frames: List[Path], sources, p, out_dir: Path, stem: str,
              pattern: str, log, emit=None) -> Dict:
    """一个滤镜组的 CFA drizzle：写 {stem}_R/_G/_B.xisf 并返回统计量"""
    import cfa
    import pws
    from drizzle import plan_output_rot, _xy_rows, _scatter_band

    t0 = time.perf_counter()
    n = len(frames)
    crop = int(getattr(p, 'crop', 0) or 0)
    center = pws.resolve_center(frames, crop, None)
    star_n = max(20, min(int(getattr(p, 'n_star', 120) or 120), 200))
    fs = pws.prepare_frames(
        frames, ';'.join(str(s) for s in sources), crop, center,
        getattr(p, 'limit', None), str(getattr(p, 'frames_dir', '') or ''),
        bool(getattr(p, 'keep_frames', False)),
        int(getattr(p, 'bgdeg', 0) or 0), int(getattr(p, 'bg_r', 0) or 0),
        star_n=star_n)
    h, w = fs.shape
    n = int(fs.n)                      # --limit 会少读，实际帧数以 fs 为准
    log(f'[CFA-DZ] 读帧完成 {time.perf_counter() - t0:.1f}s  {w}×{h}  {n} 帧')

    # ---- 配准：只在 G 平面上量几何 ----
    t1 = time.perf_counter()
    g0 = cfa.plane(np.asarray(fs.frame(0), np.float32), pattern, 'G')
    stars = pws.build_star_table(g0, n_star=star_n)
    from drizzle import measure_shifts
    sh = measure_shifts(fs, stars, ref=0, log=log, guess=None)
    dx = np.asarray(sh['dx'], np.float64)
    dy = np.asarray(sh['dy'], np.float64)
    th = np.asarray(sh['theta'], np.float64)
    kk = np.asarray(sh.get('scale', np.ones(n)), np.float64)
    dx, dy, th, kk = (a[:n] for a in (dx, dy, th, kk))
    # 实验：给每帧加随机亚像素抖动，用来判断格点是不是「抖动档位太少」造成的
    if int(os.environ.get('DWT_CFA_JITTER', 0) or 0):
        _rng = np.random.default_rng(12345)
        dx = dx + _rng.uniform(-0.5, 0.5, dx.size)
        dy = dy + _rng.uniform(-0.5, 0.5, dy.size)
        log('[CFA-DZ] 实验：叠加随机亚像素抖动 ±0.5px（检验格点成因）')
    # CFA drizzle 里 k≠1 会让 2×2 相位格与输出格产生拍频（大尺度格子纹）：
    #   这里固定 k=1（只做平移+旋转重采样），与 PI 的 Bayer drizzle 口径一致。
    if str(os.environ.get('DWT_CFA_K', '1')) != '0':
        kk = np.ones_like(kk)
    ok = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(th)
    log(f'[CFA-DZ] 配准 {time.perf_counter() - t1:.1f}s  有效 {int(ok.sum())}/{n}')

    # ---- 权重 C / S / T ----
    fw = getattr(fs, 'fwhms', None)
    fw = np.full(n, np.nan) if fw is None else np.asarray(fw, np.float64)[:n].copy()
    okf = np.isfinite(fw)
    # 「全帧同值」= 读帧阶段没量出来（拟合集体失败会退回 guess），按缺失去现算
    if okf.sum() >= 3 and float(np.nanmax(fw[okf]) - np.nanmin(fw[okf])) < 1e-3:
        okf[:] = False
    if okf.sum() < max(3, n // 2):
        # 原始 CFA 帧上是棋盘格，星点拟合必然失败：先只把 G 平面拆出来再量
        log('[CFA-DZ] 逐帧量清晰度（在 G 平面上量，避开拜耳棋盘格）')
        for i in range(n):
            try:
                gi = cfa.plane(np.asarray(fs.frame(i), np.float32), pattern, 'G')
                fw[i] = pws.measure_fwhm(gi, [(int(a), int(b)) for a, b in stars],
                                         fwhm_guess=6.0)
            except Exception:
                fw[i] = np.nan
        okf = np.isfinite(fw)
    fmed = float(np.median(fw[okf])) if okf.any() else 6.0
    C = np.where(okf, fmed / np.maximum(fw, 1e-9), 1.0)
    sig = np.asarray(fs.sigma, np.float64)
    S = np.median(sig) / np.maximum(sig, 1e-30)
    log(f'[CFA-DZ] 清晰度 C {C.min():.3f}~{C.max():.3f}  信噪比 S {S.min():.3f}~{S.max():.3f}'
        f'  FWHM 中位 {fmed:.2f}px')

    # ---- R 场：用 G 参考像算一张空间权重图，三平面共用 ----
    sigma_psf = fmed / 2.3548
    try:
        R, rinfo = pws.build_r_field(
            g0, sigma_psf,
            r_floor=float(getattr(p, 'r_floor', 0.0) or 0.0),
            r_mode=str(getattr(p, 'r_mode', 'max') or 'max'),
            r_win=float(getattr(p, 'r_win', 0.0) or 0.0),
            r_k=float(getattr(p, 'r_k', 3.0) or 3.0),
            star_grow=float(getattr(p, 'star_grow_pct', 0.0) or 0.0) * sigma_psf)
    except Exception as e:                     # R 场算不出来不该毁掉整次叠加
        log(f'[CFA-DZ] R 场计算失败，按 R=0.5 均匀处理：{e}')
        R = np.full((h, w), 0.5, np.float32)
        rinfo = {}

    # 实验开关：DWT_CFA_R=0 关闭 R 场（均匀 0.5），用来隔离棋盘纹来源
    if str(os.environ.get('DWT_CFA_R', '1')) == '0':
        R = np.full((h, w), 0.5, np.float32)
    A = float(getattr(p, 'A', 20.0))
    B = float(getattr(p, 'B', 2.0))
    rej_k = float(getattr(p, 'rej_k', 3.5) or 0.0)
    s = max(1, int(getattr(p, 'cfa_drizzle_scale', 2) or 2))
    pixfrac = 1.0                              # 硬约束：铺满 s×s
    if abs(float(getattr(p, 'drizzle_pix', 1.0) or 1.0) - 1.0) > 1e-6:
        log('[CFA-DZ] pixfrac 已强制 1.0：CFA drizzle 必须让每个像素铺满 s×s 输出格，'
            '否则会留空洞/棋盘格')

    plan = plan_output_rot(fs.shape, dx, dy, th, float(s), k=kk)
    H, W = int(plan['H']), int(plan['W'])
    log(f"[CFA-DZ] 画布 {W}×{H}  倍率 {s}×  pixfrac {pixfrac}")

    # R 场算在本机网格（h,w）上，输出画布是 s 倍且带 ox/oy 偏移：
    #   按最近邻映射到输出网格再取带，避免形状不匹配。
    _yy = np.clip(((np.arange(H) - plan['oy']) / s).astype(int), 0, h - 1)
    _xx = np.clip(((np.arange(W) - plan['ox']) / s).astype(int), 0, w - 1)
    R_out = np.asarray(R, np.float32)[np.ix_(_yy, _xx)]
    colmap = _colmap(pattern)
    allrows = np.arange(w)[None, :] % 2
    acc = np.zeros((3, H, W), np.float32)
    wsum = np.zeros((3, H, W), np.float32)
    rej_hit = rej_tot = 0.0
    rows_per_band = max(64, min(256, int(96e6 / max(W, 1) / max(n, 1))))
    t2 = time.perf_counter()
    # 逐帧、逐相位的中位校正：CFA 平场残差让四个相位的中位不同（实测 G1-G2 差 0.4%），
    # 2x 散射 drizzle 会把它们分别送进相邻输出像素，形成 2 像素棋盘纹。
    _pfac = np.ones((n, 4), np.float64)
    try:
        for _i in range(n):
            _src = getattr(fs, 'mm', None)
            _d = np.asarray(_src[_i, ::2, ::2] if _src is not None else
                            np.asarray(fs.frame(_i))[::2, ::2], np.float32)
            _yy = (np.arange(_d.shape[0]) % 2)[:, None]
            _xx = (np.arange(_d.shape[1]) % 2)[None, :]
            _ms = [float(np.median(_d[(_yy == (p // 2)) & (_xx == (p % 2))]))
                   for p in range(4)]
            _t = float(np.median([m for m in _ms if m > 0]))
            if _t > 0:
                _pfac[_i] = [(_t / m) if m > 0 else 1.0 for m in _ms]
        log('[CFA-DZ] 相位平衡：4 个 CFA 相位的逐帧中位已对齐（消除 2 像素棋盘纹）')
    except Exception as _e:
        log(f'[CFA-DZ] 相位平衡跳过：{type(_e).__name__}: {_e}')
    _shared = {'n': n, 'h': h, 'w': w, 'W': W, 'H': H, 's': s, 'pixfrac': pixfrac,
               'dx': dx, 'dy': dy, 'th': th, 'kk': kk, 'ok': ok,
               'C': C, 'S': S, 'A': A, 'B': B, 'rej_k': rej_k,
               'R_out': R_out, 'plan': plan, 'colmap': colmap, 'allrows': allrows,
               'mmpath': str(fs.path) if getattr(fs, 'path', None) else '',
               'off': np.asarray(getattr(fs, 'off', np.zeros(n)), np.float64),
               'bgcoef': getattr(fs, 'bgcoef', None),
               'bgdeg': int(getattr(fs, 'bgdeg', 0) or 0),
               'bgrcoef': getattr(fs, 'bgrcoef', None),
               'bgrdeg': int(getattr(fs, 'bgrdeg', 0) or 0),
               'bgshape': tuple(getattr(fs, 'bgshape', (h, w))), 'pfac': _pfac}
    rows_per_band = int(os.environ.get('DWT_CFA_BAND', rows_per_band) or rows_per_band)
    bands = [(Y0, min(H, Y0 + rows_per_band)) for Y0 in range(0, H, rows_per_band)]
    if _shared['mmpath']:
        import multiprocessing as mp
        jobs = max(1, min(6, (os.cpu_count() or 4) - 2))
        # 父进程先让出 memmap，子进程只读打开（Windows 上 w+ 句柄不能共享）
        if getattr(fs, 'mm', None) is not None:
            try:
                fs.mm.flush()
            except Exception:
                pass
            del fs.mm
            fs.mm = None
        with mp.Pool(jobs, initializer=_band_init, initargs=(_shared,)) as pool:
            for k, (Y0, acc_b, wsum_b, rh, rt) in enumerate(
                    pool.imap_unordered(_band_one, bands, chunksize=1), 1):
                acc[:, Y0:Y0 + acc_b.shape[1]] = acc_b
                wsum[:, Y0:Y0 + wsum_b.shape[1]] = wsum_b
                rej_hit += rh
                rej_tot += rt
                if emit:
                    emit('CFA-DZ', k / max(len(bands), 1), f'{k}/{len(bands)} 带')
    else:
        _band_init(_shared)
        for k, (Y0, Y1) in enumerate(bands, 1):
            Y0, acc_b, wsum_b, rh, rt = _band_one(Y0, Y1)
            acc[:, Y0:Y0 + acc_b.shape[1]] = acc_b
            wsum[:, Y0:Y0 + wsum_b.shape[1]] = wsum_b
            rej_hit += rh
            rej_tot += rt
            if emit:
                emit('CFA-DZ', k / max(len(bands), 1), f'{k}/{len(bands)} 带')
    log(f'[CFA-DZ] 累加+排异 {time.perf_counter() - t2:.1f}s  '
        f'剔除率 {100.0 * rej_hit / max(rej_tot, 1.0):.3f}%')

    out = np.stack([acc[c] / np.maximum(wsum[c], 1e-9) for c in range(3)])
    for c in range(3):
        out[c][wsum[c] <= 1e-9] = np.nan
    # 天空加性平移补回（prepare_frames 扣过），三平面用同一常数，避免偏色
    out += np.float32(fs.sky0)

    out_dir.mkdir(parents=True, exist_ok=True)
    mode = (f'CFA Drizzle {s}×（pixfrac {pixfrac}，不解马赛克；'
            f'逐平面排异 k={rej_k:g}；A={A:g} B={B:g}）')
    for c, ch in enumerate('RGB'):
        pws.save_linear_xisf(out_dir / f'{stem}_{ch}.xisf',
                             np.ascontiguousarray(out[c], np.float32),
                             fs.metas, mode)
    try:
        pws._merge_cfa_rgb(out_dir, stem, log)
    except Exception as e:
        log(f'[CFA-DZ] RGB 合成失败（三个单通道成品仍在）：{e}')

    g_out = np.nan_to_num(out[1], nan=0.0)
    st_out = pws.build_star_table(g_out, n_star=min(star_n, 120))
    fwhm_out = pws.measure_fwhm(g_out, [(int(a), int(b)) for a, b in st_out], fwhm_guess=6.0)
    if not np.isfinite(fwhm_out):
        fwhm_out = float('nan')
    try:
        fs.close(keep=bool(getattr(p, 'keep_frames', False)))
    except Exception:
        pass
    log(f'[CFA-DZ] 成品星点 FWHM {fwhm_out:.2f}px（图上 {fwhm_out * s:.2f}px @×{s}）'
        f'  单帧中位 {fmed:.2f}px')
    return {'n': n, 'shape': (H, W), 'fwhm_med': fmed, 'fwhm_out': fwhm_out,
            'rej_rate': rej_hit / max(rej_tot, 1.0), 'stack': None,
            'stem': stem, 'out_dir': str(out_dir), 'cfa_drizzle': True}
