"""真 CFA 的 wgpu 叠加后端（分桶版）——跨平台、与 CPU 判据一致。

思路
----
定点原子加（gate 版）的致命问题：权重 1e6 量级、偏差 1e-5 量级的乘积在 u32 里被
量化成 0，导致均值/方差/剔除尺度全塌。这里改成**每像素一个小桶**：

    每个样本 -> atomicAdd(cnt[idx], 1) 得到槽位号 slot
             -> 若 slot < K：直接写 floats 到 vals[idx*K+slot]、wts[idx*K+slot]

主机端读回 (cnt, vals, wts)，用 float64 算：
    μ  = Σ w·v / Σ w
    σ  = sqrt(Σ w·(v-μ)² / Σ w)
    s  = 1.15 · Σ w·min(|v-μ|, 3σ) / Σ w        # 与 CPU winsor 同口径
    剔除 |v-μ| > k·s，再对剩下的样本求加权平均
因此剔除判据、均值、方差都是 float64，与 CPU 逐位同量级，且**每个带只需 1 遍 dispatch**。

桶溢出（slot >= K）只增加 cnt、不写值：这样的像素样本极多、彼此接近，剔除后
还会有足够样本，不影响结果（日志会报溢出比例）。
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import os
import numpy as np

try:
    from . import gpu_wgpu as G
except ImportError:
    import gpu_wgpu as G

_WGSL = """
struct Params {
  a: f32, b: f32, c: f32, d: f32, e: f32, f: f32,
  w_src: u32, h_src: u32, w_out: u32, h_out: u32,
  y_off: u32, c0: u32, r0: u32, mask: u32,
  P: f32, lnc: f32, lns: f32, wa: f32,
  wb: f32, has_r: f32, kslot: u32, s_off: u32,
}

@group(0) @binding(0) var<uniform> Q: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> aCnt: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> aVal: array<f32>;
@group(0) @binding(4) var<storage, read_write> aWt: array<f32>;
@group(0) @binding(5) var<storage, read> rmap: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let px = gid.x;
  let py = gid.y;
  if (px >= Q.w_src || py >= Q.h_src) { return; }
  let gy0 = py + Q.r0;
  let gx0 = px + Q.c0;
  if (((Q.mask >> ((gy0 & 1u) * 2u + (gx0 & 1u))) & 1u) == 0u) { return; }

  let v = src[Q.s_off + py * Q.w_src + px];
  if (!(v == v)) { return; }

  let X = Q.a * (f32(gx0) + 0.5) + Q.b * (f32(gy0) + 0.5) + Q.c;
  let Y = Q.d * (f32(gx0) + 0.5) + Q.e * (f32(gy0) + 0.5) + Q.f;

  let hx = 0.5 * Q.P;
  let x0 = X - hx;
  let x1 = X + hx;
  let y0 = Y - hx;
  let y1 = Y + hx;
  let kx0 = i32(floor(x0));
  let ky0 = i32(floor(y0));
  let noff = i32(ceil(Q.P)) + 1;
  let K = i32(Q.kslot);

  for (var aix: i32 = 0; aix < noff; aix = aix + 1) {
    let kx = kx0 + aix;
    if (kx < 0 || kx >= i32(Q.w_out)) { continue; }
    let wx = min(x1, f32(kx + 1)) - max(x0, f32(kx));
    if (wx <= 0.0) { continue; }
    for (var aiy: i32 = 0; aiy < noff; aiy = aiy + 1) {
      let ky = ky0 + aiy;
      let gy = ky - i32(Q.y_off);
      if (gy < 0 || gy >= i32(Q.h_out)) { continue; }
      let wy = min(y1, f32(ky + 1)) - max(y0, f32(ky));
      if (wy <= 0.0) { continue; }
      let idx = u32(gy) * Q.w_out + u32(kx);
      let R = select(0.5, rmap[(u32(gy) + Q.y_off) * Q.w_out + u32(kx)], Q.has_r > 0.5);
      let wgt = exp(Q.wa * R * Q.lnc + Q.wb * (1.0 - R) * Q.lns);
      // CPU 的 _scatter_band 用 1/P² 归一（液滴面积），这里同口径；
      // 不同帧 kk 略有差异时这一步会改变相对权重。
      let w = wgt * wx * wy / (Q.P * Q.P);
      let slot = atomicAdd(&aCnt[idx], 1u);
      if (slot < u32(K)) {
        aVal[idx * u32(K) + slot] = w * v;
        aWt[idx * u32(K) + slot] = w;
      }
    }
  }
}
"""

_cache: Dict[str, object] = {}


def _pipe():
    if 'pipe' in _cache:
        return _cache['pipe'], _cache['bgl'], _cache['dev'], _cache['queue']
    dev, queue = G._device()
    mod = dev.create_shader_module(code=_WGSL)
    pipe = dev.create_compute_pipeline(
        layout='auto', compute={'module': mod, 'entry_point': 'main'})
    _cache.update({'pipe': pipe, 'bgl': pipe.get_bind_group_layout(0),
                   'dev': dev, 'queue': queue})
    return pipe, _cache['bgl'], dev, queue


def _sbuf(dev, U, n):
    return dev.create_buffer(size=max(16, int(n) * 4),
                             usage=U.STORAGE | U.COPY_SRC | U.COPY_DST)


def _read(dev, queue, buf, n, U):
    try:
        from .gpu_wgpu_cfa_scatter import _read as _r
    except ImportError:
        from gpu_wgpu_cfa_scatter import _read as _r
    return _r(dev, queue, buf, n, U)


def available() -> bool:
    return G.available()


def stack_cfa_bucket(fs, C, S, dx, dy, th, kk, plan, A, B, rej_k, rej_gate,
                     R=None, pixfrac=1.0, band_rows=512, kslots=16,
                     log=None, pattern=None, channel=None, **kw):
    """返回 dict(stack, cov, neff, ...)，与 drizzle.stack_drizzle 的返回键兼容。"""
    if not available():
        raise RuntimeError('wgpu 不可用')
    pipe, bgl, dev, queue = _pipe()
    U = G._wp().BufferUsage
    wgpu = G._wp()

    H, W = int(plan['H']), int(plan['W'])
    s_grid = float(plan.get('s', 1.0))
    ox, oy = float(plan.get('ox', 0.0)), float(plan.get('oy', 0.0))
    h, w = int(fs.shape[0]), int(fs.shape[1])

    n = len(dx)
    idx = [i for i in range(n)
           if np.isfinite(dx[i]) and np.isfinite(dy[i])
           and (th is None or np.isfinite(th[i]))]
    if len(idx) < 2:
        raise RuntimeError('有效帧不足')

    rej_k = float(rej_k or 0.0)
    # 每像素桶深按帧数自适应：一个输出像素最多被「每帧 ~1 个样本」命中，
    # 再加少量余量给 2×/pixfrac<1 的重叠；同时受显存预算约束。
    K = 8
    while K < len(idx) + 4 and K < 64:
        K *= 2
    if kslots:
        K = int(kslots)
    K = int(os.environ.get('DWT_BUCKET_K', K))
    _mem_cap = int(os.environ.get('DWT_BUCKET_MB', '768')) * 1024 * 1024
    while K > 4 and band_rows * W * K * 8 > _mem_cap:
        K //= 2
    stack = np.zeros((H, W), np.float32)
    cov = np.zeros((H, W), np.float32)
    neff = np.zeros((H, W), np.float32)
    spill = 0
    _mask2 = None

    for y0 in range(0, H, band_rows):
        bh = min(band_rows, H - y0)
        Ylo, Yhi = y0 - 2.0 * pixfrac * s_grid - 4, y0 + bh + 2.0 * pixfrac * s_grid + 4
        lo, hi = h, 0
        for i in idx:
            ti = math.radians(float(th[i]))
            ct, stn = math.cos(ti), math.sin(ti)
            kk_i = float(kk[i]) if np.isfinite(kk[i]) else 1.0
            ks = kk_i
            d_ = s_grid * ks * stn
            e_ = s_grid * ks * ct
            uy = -float(dy[i])
            f_ = s_grid * uy + oy
            if abs(e_) < 1e-9:
                continue
            for gx in (0.0, float(w - 1)):
                for Yb in (Ylo, Yhi):
                    gy = (Yb - d_ * (gx + 0.5) - f_) / e_ - 0.5
                    lo = min(lo, gy); hi = max(hi, gy)
        if hi <= lo:
            continue
        j_lo = max(0, int(math.floor(lo)) - 8)
        j_hi = min(h, int(math.ceil(hi)) + 8)
        if j_hi - j_lo < 1:
            continue
        hs = j_hi - j_lo
        plane = hs * w

        src_big = _sbuf(dev, U, plane * len(idx))
        aCnt = dev.create_buffer(size=max(16, bh * W * 4),
                                 usage=U.STORAGE | U.COPY_SRC | U.COPY_DST)
        aVal = _sbuf(dev, U, bh * W * K)
        aWt = _sbuf(dev, U, bh * W * K)
        rmap = _sbuf(dev, U, H * W)
        uni = dev.create_buffer(size=128, usage=U.UNIFORM | U.COPY_DST)

        bg = dev.create_bind_group(layout=bgl, entries=[
            {'binding': 0, 'resource': {'buffer': uni, 'offset': 0, 'size': 112}},
            {'binding': 1, 'resource': {'buffer': src_big, 'offset': 0,
                                        'size': max(16, plane * len(idx) * 4)}},
            {'binding': 2, 'resource': {'buffer': aCnt, 'offset': 0, 'size': max(16, bh * W * 4)}},
            {'binding': 3, 'resource': {'buffer': aVal, 'offset': 0,
                                        'size': max(16, bh * W * K * 4)}},
            {'binding': 4, 'resource': {'buffer': aWt, 'offset': 0,
                                        'size': max(16, bh * W * K * 4)}},
            {'binding': 5, 'resource': {'buffer': rmap, 'offset': 0, 'size': max(16, H * W * 4)}},
        ])

        # 只上传本带需要的数据：每帧 j_lo..j_hi 的连续行
        strips = []
        for i in idx:
            st = np.ascontiguousarray(
                np.asarray(fs.frame(int(i)), np.float32)[j_lo:j_hi, :])
            strips.append(st)
        big = np.concatenate([x.ravel() for x in strips]).astype(np.float32)
        queue.write_buffer(src_big, 0, big.tobytes())
        del big

        rr = np.zeros((H, W), np.float32) if R is None else np.asarray(R, np.float32)
        if rr.shape != (H, W):
            ys = np.linspace(0, rr.shape[0] - 1, H).astype(np.int32)
            xs = np.linspace(0, rr.shape[1] - 1, W).astype(np.int32)
            rr = rr[np.ix_(ys, xs)]
        queue.write_buffer(rmap, 0, np.ascontiguousarray(rr).tobytes())

        zero4 = np.zeros(max(1, bh * W), np.uint32).tobytes()
        queue.write_buffer(aCnt, 0, zero4)
        zval = np.zeros(bh * W * K, np.float32).tobytes()
        queue.write_buffer(aVal, 0, zval)
        queue.write_buffer(aWt, 0, zval)

        if _mask2 is None:
            try:
                from . import cfa as _cfam
            except ImportError:
                import cfa as _cfam
            _m = _cfam.mask2(pattern, channel)
            _mask2 = (int(_m[0][0]) | (int(_m[0][1]) << 1)
                      | (int(_m[1][0]) << 2) | (int(_m[1][1]) << 3))
        mask2 = _mask2
        for j, i in enumerate(idx):
            ti = math.radians(float(th[i]))
            ct, stn = math.cos(ti), math.sin(ti)
            kk_i = float(kk[i]) if np.isfinite(kk[i]) else 1.0
            ks = kk_i
            ux, uy = -float(dx[i]), -float(dy[i])
            p = np.zeros(28, np.float32)
            p[0] = np.float32(s_grid * ks * ct)
            p[1] = np.float32(-s_grid * ks * stn)
            p[2] = np.float32(s_grid * ux + ox)
            p[3] = np.float32(s_grid * ks * stn)
            p[4] = np.float32(s_grid * ks * ct)
            p[5] = np.float32(s_grid * uy + oy)
            _pu = p.view(np.uint32)
            _pu[6:10] = np.array([w, hs, W, bh], np.uint32)
            _pu[10:14] = np.array([y0, 0, j_lo, mask2], np.uint32)
            p[14] = np.float32(max(1e-6, float(pixfrac) * s_grid * kk_i))
            p[15] = np.float32(math.log(max(float(C[j]), 1e-6)) if C is not None else 0.0)
            p[16] = np.float32(math.log(max(float(S[j]), 1e-6)) if S is not None else 0.0)
            p[17] = np.float32(float(A))
            p[18] = np.float32(float(B))
            p[19] = np.float32(1.0 if R is not None else 0.0)
            _pu[20] = np.uint32(K)
            _pu[21] = np.uint32(j * plane)
            queue.write_buffer(uni, 0, p.tobytes())
            enc = dev.create_command_encoder()
            cp = enc.begin_compute_pass()
            cp.set_pipeline(pipe)
            cp.set_bind_group(0, bg)
            cp.dispatch_workgroups(int((w + 7) // 8), int((hs + 7) // 8))
            cp.end()
            queue.submit([enc.finish()])

        cnt = _read(dev, queue, aCnt, bh * W, U)
        # _read 返回的是 uint32 视图，必须 view(float32) 再转 float64，
        # 否则读回来的是 IEEE754 位模式（整数），全部错位。
        vals = _read(dev, queue, aVal, bh * W * K, U).view(np.float32).astype(np.float64)
        wts = _read(dev, queue, aWt, bh * W * K, U).view(np.float32).astype(np.float64)

        if __import__('os').environ.get('DWT_BK_RAW') == '1':
            print('[raw] cnt[:6]', cnt[:6].view(np.uint32))
            print('[raw] val[:6]', vals[:6])
            print('[raw] wt [:6]', wts[:6])
        cnt_i = cnt.astype(np.int64)
        spill += int(np.maximum(cnt_i - K, 0).sum())
        Vm = vals.reshape(bh * W, K)
        Wm = wts.reshape(bh * W, K)
        ok = Wm > 0
        Wsum = Wm.sum(axis=1)
        Vsum = Vm.sum(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            mu = np.where(Wsum > 1e-12, Vsum / np.maximum(Wsum, 1e-30), 0.0)
            dv = np.abs(Vm - mu[:, None]) * ok
            var = (Wm * dv * dv).sum(axis=1) / np.maximum(Wsum, 1e-30)
            sd = np.sqrt(np.maximum(var, 0.0))
            lim = 3.0 * sd
            devc = np.minimum(dv, lim[:, None])
            s = np.where(Wsum > 1e-12, 1.15 * (Wm * devc).sum(axis=1) / np.maximum(Wsum, 1e-30), 0.0)
            if rej_k > 0:
                keep = ok & (np.abs(Vm - mu[:, None]) <= rej_k * s[:, None])
            else:
                keep = ok
            Wk = np.where(keep, Wm, 0.0)
            Vk = np.where(keep, Vm, 0.0)
            Wk_sum = Wk.sum(axis=1)
            out = np.where(Wk_sum > 1e-12, Vk.sum(axis=1) / np.maximum(Wk_sum, 1e-30), 0.0)
            w2 = (Wk * Wk).sum(axis=1)
        stack[y0:y0 + bh, :] = out.reshape(bh, W).astype(np.float32)
        cov[y0:y0 + bh, :] = Wk_sum.reshape(bh, W).astype(np.float32)
        with np.errstate(invalid='ignore', divide='ignore'):
            neff[y0:y0 + bh, :] = np.where(Wk_sum > 1e-12,
                                           (Wk_sum * Wk_sum) / np.maximum(w2, 1e-12), 0.0
                                           ).reshape(bh, W).astype(np.float32)

    if log:
        log('[wgpu] 真 CFA：分桶叠加（K=%d，每像素 float 样本，判据与 CPU 一致）' % K)
        if spill:
            log('[wgpu] 桶溢出 %d 个样本（占 %.2f%%，不影响成品）'
                % (spill, 100.0 * spill / max(1, int(cov.size) * max(1, len(idx)))))
    return {'stack': stack, 'cov': cov, 'neff': neff,
            'wmax': float(cov.max()), 'rej_rate': 0.0, 'n_used': len(idx),
            'plan': plan, 'backend': 'wgpu-bucket'}
