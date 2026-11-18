# -*- coding: utf-8 -*-
"""GPU 版 drizzle 带核 —— 与 `_drizzle_band` 同语义，逐位可比。"""
import math

import numpy as np

import gpu_dz as G

cp = G.cp
ANCHOR_N = 48


def band(ctx, Y0, Y1):
    fs = ctx['fs']
    R_out = ctx['R_out']
    C, S = ctx['C'], ctx['S']
    T_fl, fp = ctx.get('T'), float(ctx.get('fp', 0.0) or 0.0)
    if T_fl is not None and fp:
        _T = np.asarray(T_fl, np.float64)
        _T = np.where(np.isfinite(_T) & (_T > 0), _T, 1.0)
        lt_all_np = np.log(np.maximum(_T, 1e-12))
    else:
        lt_all_np = None
    dx, dy, plan = ctx['dx'], ctx['dy'], ctx['plan']
    A, B = ctx['A'], ctx['B']
    rej_k, rej_gate = ctx['rej_k'], ctx['rej_gate']
    sky, pixfrac = ctx['sky'], ctx['pixfrac']
    sw = float(ctx.get('sharp_w', 0.0) or 0.0)
    rot, theta, scale = ctx['rot'], ctx['theta'], ctx['scale']
    h, w = ctx['native_shape']
    s, W = ctx['s'], ctx['W']
    idx = ctx['idx']
    dist = ctx.get('dist')
    bh = Y1 - Y0
    n = len(idx)

    num = cp.zeros((n, bh, W), cp.float32)
    cov = cp.zeros((n, bh, W), cp.float32)
    for k, i in enumerate(idx):
        if theta is not None:
            th = float(theta[i])
        elif rot is not None:
            th = float(np.degrees(math.atan2(rot[i][1, 0], rot[i][0, 0])))
        else:                      # 纯平移桩（测试用）
            th = 0.0
        kk = 1.0 if scale is None else float(scale[i])
        if not np.isfinite(kk):
            kk = 1.0
        dd = 0.0 if dist is None else float(dist[i])
        if not np.isfinite(dd):
            dd = 0.0
        Pi = max(1e-6, float(pixfrac) * s * kk)
        ux, uy = -float(dx[i]), -float(dy[i])
        j0, j1 = ctx['rows_fn'](plan, th, uy, Y0, Y1, h, w, Pi, kk, dd)
        if j1 <= j0:
            continue
        strip = cp.asarray(np.asarray(fs.strip(i, j0, j1), np.float32))
        X, Y = G._xy_gpu(j0, j1, w, s, plan['ox'], plan['oy'], th, ux, uy, kk, dd, h)
        num[k], cov[k] = G._scatter_gpu(strip, X, Y, Pi, W, Y0, Y1)
        del strip, X, Y

    R = cp.asarray(np.ascontiguousarray(R_out[Y0:Y1], np.float32))
    good = cov > 1e-6
    goodcol = good.any(axis=0)
    v = cp.where(good, num / cp.maximum(cov, 1e-30), cp.nan).astype(cp.float32)
    rej_num = rej_den = 0.0
    if rej_k > 0:
        if n > ANCHOR_N:
            sel = cp.linspace(0, n - 1, ANCHOR_N).astype(cp.int64)
            vs = v[sel]
        else:
            sel = None
            vs = v
        anchor = cp.zeros((bh, W), cp.float32)
        if bool(goodcol.any()):
            an = cp.nanmedian(cp.where(goodcol[None], vs, cp.nan), axis=0)
            anchor = cp.where(goodcol, cp.where(cp.isfinite(an), an, 0.0), 0.0).astype(cp.float32)
        _mode = str(ctx.get('rej_mode', 'winsor'))
        _s_pix = (cp.nanstd(v, axis=0).astype(cp.float32)
                  if _mode == 'winsor' else None)
        dev = cp.abs(v - anchor[None])
        mad = cp.zeros((bh, W), cp.float32)
        if bool(goodcol.any()):
            dv = dev if sel is None else dev[sel]
            md = cp.nanmedian(cp.where(goodcol[None], dv, cp.nan), axis=0)
            mad = cp.where(goodcol, 1.4826 * cp.where(cp.isfinite(md), md, 0.0), 0.0).astype(cp.float32)
        if _mode == 'winsor':
            # 与 CPU 版同口径（含 1/9 抽样）：按值标准差截尾，再求 |偏差| 分块均值 ×1.15
            _bk = int(ctx.get('rej_block', 64))
            _st = 3
            _sub = dev[:, ::_st, ::_st]
            _sp = cp.asarray(_s_pix[::_st, ::_st], cp.float32)
            _gc = cp.asarray(goodcol[::_st, ::_st])
            _hs, _ws = _sub.shape[1], _sub.shape[2]
            _ny = (bh + _bk - 1) // _bk
            _nx = (W + _bk - 1) // _bk
            _vals = cp.zeros((_ny, _nx), cp.float32)
            for _iy in range(_ny):
                _y0 = _iy * _bk
                _y1 = min(bh, (_iy + 1) * _bk)
                _sy0 = (_y0 + _st - 1) // _st
                _sy1 = min(_hs, (_y1 + _st - 1) // _st)
                if _sy1 <= _sy0:
                    continue
                for _ix in range(_nx):
                    _x0 = _ix * _bk
                    _x1 = min(W, (_ix + 1) * _bk)
                    _sx0 = (_x0 + _st - 1) // _st
                    _sx1 = min(_ws, (_x1 + _st - 1) // _st)
                    if _sx1 <= _sx0:
                        continue
                    _sv = _sp[_sy0:_sy1, _sx0:_sx1]
                    _sv = _sv[_sv > 0]
                    _lim = float(cp.asnumpy(_sv.mean())) if _sv.size else 0.0
                    _lim *= max(2.0, float(rej_k))
                    _blk = _sub[:, _sy0:_sy1, _sx0:_sx1]
                    if _lim > 0:
                        _blk = cp.minimum(_blk, cp.float32(_lim))
                    _dv = _blk[:, _gc[_sy0:_sy1, _sx0:_sx1]]
                    if _dv.size:
                        _vals[_iy, _ix] = cp.float32(1.15) * cp.nanmean(_dv)
            mad = cp.kron(_vals, cp.ones((_bk, _bk), cp.float32))[:bh, :W]
            mad = cp.where(goodcol, mad, cp.float32(0.0)).astype(cp.float32)
        if _mode == 'mad':
            try:
                _bk = int(ctx.get('rej_block', 64))
                _ds = dev[_sel] if n > ANCHOR_N else dev
                _m2 = cp.zeros((bh, W), cp.float32)
                for _y0 in range(0, bh, _bk):
                    _y1 = min(bh, _y0 + _bk)
                    for _x0 in range(0, W, _bk):
                        _x1 = min(W, _x0 + _bk)
                        _blk = _ds[:, _y0:_y1, _x0:_x1]
                        _a = cp.nanmedian(_blk, axis=0)
                        _s2 = float(cp.nanmedian(_a)) if _a.size else 0.0
                        _m2[_y0:_y1, _x0:_x1] = 1.4826 * _s2
                mad = cp.where(goodcol, _m2, 0.0).astype(cp.float32)
            except Exception:
                pass
        sig = cp.maximum(anchor - cp.float32(sky), 0.0)
        core = (R > 0.5) & (sig > 0)
        m = 0.0
        if bool(core.any()):
            d = dev[:, core] / sig[core][None]
            cnt = cp.isfinite(d).sum(axis=1)
            if bool((cnt >= 5).any()):
                # CuPy 没有 nanpercentile；这块数据量不大（n×核心区像素），
                # 搬回 CPU 用 numpy 算，保证与 CPU 版**同一套定义**。
                # 实测比在 GPU 上 cp.sort 再插值快（46.6 s vs 49.0 s / 60 帧）。
                dd_ = cp.asnumpy(d[cnt >= 5])
                r = np.nanpercentile(dd_, 99, axis=1)
                r = r[np.isfinite(r)]
                if r.size:
                    m = float(np.median(r))
        thr = cp.float32(rej_k) * mad + cp.float32(m) * (R * sig)
        bad = (dev > thr[None]) & good
        nbad = bad.sum(axis=0)
        expected = n * math.erfc(rej_k / 2.0 ** 0.5)
        bad &= (nbad - expected <= max(1.0, rej_gate * n))[None]
        rej_num += float(bad.sum())
        rej_den += float(good.sum())
        cov = cp.where(bad, cp.float32(0.0), cov)
    v = cp.where(cov > 1e-6, num / cp.maximum(cov, 1e-30), 0.0).astype(cp.float32)
    lc = cp.log(cp.maximum(cp.asarray(C, cp.float32), cp.float32(1e-12)))
    ls = cp.log(cp.maximum(cp.asarray(S, cp.float32), cp.float32(1e-12)))
    lt_all = None if lt_all_np is None else cp.asarray(lt_all_np, cp.float32)
    nr = cp.zeros((bh, W), cp.float32)
    dr = cp.zeros((bh, W), cp.float32)
    d2 = cp.zeros((bh, W), cp.float32)
    mxw = cp.zeros((bh, W), cp.float32)
    base = (cp.float32(A) * R)[None] * lc[:, None, None] + \
           (cp.float32(B) * (1.0 - R))[None] * ls[:, None, None]
    if sw:
        base = base + cp.float32(sw) * lc[:, None, None]
    for k, i in enumerate(idx):
        e = base[k]
        if lt_all is not None:
            e = e + cp.float32(fp) * lt_all[i]
        wgt = cp.exp(e).astype(cp.float32)
        wc = wgt * cov[k]
        nr += wc * v[k]
        dr += wc
        d2 += wc * wc
        mxw = cp.maximum(mxw, wgt)
    live = dr > 1e-30
    safe = cp.where(live, dr, cp.float32(1.0))
    ncov = (cov > 1e-6).sum(axis=0).astype(cp.float32)
    return {'stack': cp.asnumpy(cp.where(live, nr / safe, 0.0).astype(cp.float32)),
            'neff': cp.asnumpy(cp.where(live, (dr * dr) / cp.maximum(d2, 1e-30), 0.0).astype(cp.float32)),
            'cov': cp.asnumpy(dr.astype(cp.float32)),
            'ncov': cp.asnumpy(ncov),
            'rej_num': rej_num, 'rej_den': rej_den,
            'wmax': [float(cp.asnumpy(mxw).mean())]}
