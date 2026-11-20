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
import time
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
  lnt: f32,
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
      let w = wgt * wx * wy / (Q.P * Q.P) * exp(Q.lnt);
      let slot = atomicAdd(&aCnt[idx], 1u);
      if (slot < u32(K)) {
        aVal[idx * u32(K) + slot] = w * v;
        aWt[idx * u32(K) + slot] = w;
      }
    }
  }
}
"""


# 归约核：每个输出像素读自己的 K 个桶，算 μ/σ/winsor 尺度、剔除、出成品。
# 这一步不需要原子操作，纯读改写，放在 GPU 上做可以把主机从 7~13s 的统计里解放出来。
_REDUCE_WGSL = """
struct RU {
  n_pix: u32,
  K: u32,
  rej_k: f32,
};

@group(0) @binding(0) var<storage, read> rVals: array<f32>;
@group(0) @binding(1) var<storage, read> rWts: array<f32>;
@group(0) @binding(2) var<storage, read_write> rOut: array<f32>;
@group(0) @binding(3) var<storage, read_write> rCov: array<f32>;
@group(0) @binding(4) var<storage, read_write> rNeff: array<f32>;
@group(0) @binding(5) var<uniform> U: RU;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= U.n_pix) { return; }
  let K = U.K;
  var ws: f32 = 0.0;
  var vs: f32 = 0.0;
  for (var k: u32 = 0u; k < K; k = k + 1u) {
    let w = rWts[i * K + k];
    ws = ws + w;
    vs = vs + rVals[i * K + k];
  }
  if (ws <= 1e-20) { rOut[i] = 0.0; rCov[i] = 0.0; rNeff[i] = 0.0; return; }
  let mu = vs / ws;
  var var_: f32 = 0.0;
  for (var k: u32 = 0u; k < K; k = k + 1u) {
    let w = rWts[i * K + k];
    let d = rVals[i * K + k] - mu;
    var_ = var_ + w * d * d;
  }
  let sd = sqrt(max(var_ / ws, 0.0));
  let lim = 3.0 * sd;
  var ads: f32 = 0.0;
  for (var k: u32 = 0u; k < K; k = k + 1u) {
    let w = rWts[i * K + k];
    let d = abs(rVals[i * K + k] - mu);
    ads = ads + w * min(d, lim);
  }
  let thr = U.rej_k * 1.15 * ads / ws;
  var wk: f32 = 0.0;
  var vk: f32 = 0.0;
  var w2: f32 = 0.0;
  for (var k: u32 = 0u; k < K; k = k + 1u) {
    let w = rWts[i * K + k];
    if (w <= 0.0) { continue; }
    let v = rVals[i * K + k];
    if (U.rej_k > 0.0 && abs(v - mu) > thr) { continue; }
    wk = wk + w;
    vk = vk + v;
    w2 = w2 + w * w;
  }
  if (wk <= 1e-20) { rOut[i] = 0.0; rCov[i] = 0.0; rNeff[i] = 0.0; return; }
  rOut[i] = vk / wk;
  rCov[i] = wk;
  rNeff[i] = (wk * wk) / max(w2, 1e-20);
}
"""


def _rpipe(dev):
    key = 'rpipe'
    if key in _cache:
        return _cache[key]
    mod = dev.create_shader_module(code=_REDUCE_WGSL)
    pipe = dev.create_compute_pipeline(
        layout='auto', compute={'module': mod, 'entry_point': 'main'})
    _cache[key] = pipe
    return pipe

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
                     R=None, pixfrac=1.0, band_rows=512, kslots=None,
                     T=None, fp=1.0,
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
    # 桶深要按「液滴摊到的输出像素数」放大：2× / pixfrac<1 时一个源样本会落进
    # 最多 4 个输出像素，同一个像素因此可能收到 ~n·(pixfrac·s)² 个样本。
    _spread = max(1.0, float(pixfrac) * s_grid)
    _need = int(min(128, max(16, round(len(idx) * _spread * _spread) + 4)))
    K = 16
    while K < _need and K < 128:
        K *= 2
    if kslots:
        K = int(kslots)
    K = int(os.environ.get('DWT_BUCKET_K', K))
    # 桶深只由帧数决定（少了会溢出丢样本）；显存不够时缩短每条带的行数，
    # 而不是降低桶深——这一点在 51 帧实测里差别很大（K=16 溢出 9.35%）。
    _ = int(os.environ.get('DWT_BUCKET_MB', '768'))
    stack = np.zeros((H, W), np.float32)
    cov = np.zeros((H, W), np.float32)
    neff = np.zeros((H, W), np.float32)
    spill = 0
    _mask2 = None

    _prof = os.environ.get('DWT_BUCKET_PROF') == '1'
    _t_cache = _t_read = _t_up = _t_gpu = _t_back = _t_host = 0.0
    _cache = None
    try:
        _need = len(idx) * int(h) * int(w) * 4
        if _need <= int(os.environ.get('DWT_FRAME_CACHE_MB', '1200')) * 1024 * 1024:
            _t0 = time.perf_counter()
            _cache = [np.asarray(fs.frame(int(i)), np.float32) for i in idx]
            _t_cache = time.perf_counter() - _t0
    except Exception:
        _cache = None
    if R is not None:
        _rr = np.asarray(R, np.float32)
        if _rr.shape != (H, W):
            _ys = np.linspace(0, _rr.shape[0] - 1, H).astype(np.int32)
            _xs = np.linspace(0, _rr.shape[1] - 1, W).astype(np.int32)
            _rr = _rr[np.ix_(_ys, _xs)]
    else:
        _rr = np.zeros((H, W), np.float32)
    _rmap = None
    _t0 = time.perf_counter()
    _rmap = _sbuf(dev, U, H * W)
    queue.write_buffer(_rmap, 0, np.ascontiguousarray(_rr).tobytes())
    _t_up += time.perf_counter() - _t0
    _bud_mb = int(os.environ.get('DWT_BUCKET_MB', '768'))
    _rows_by_mem = max(32, int(_bud_mb * 1024 * 1024 // max(1, W * K * 8)))
    band_rows = int(min(band_rows, _rows_by_mem))
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
            {'binding': 5, 'resource': {'buffer': _rmap, 'offset': 0, 'size': max(16, H * W * 4)}},
        ])

        # 只上传本带需要的数据：每帧 j_lo..j_hi 的连续行
        _t0 = time.perf_counter()
        if _cache is not None:
            strips = [np.ascontiguousarray(x[j_lo:j_hi, :]) for x in _cache]
        else:
            strips = [np.ascontiguousarray(
                np.asarray(fs.frame(int(i)), np.float32)[j_lo:j_hi, :]) for i in idx]
        _t_read += time.perf_counter() - _t0
        big = np.concatenate([x.ravel() for x in strips]).astype(np.float32)
        _t0 = time.perf_counter()
        queue.write_buffer(src_big, 0, big.tobytes())
        _t_up += time.perf_counter() - _t0
        del big

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
            _tv = float(T[j]) if T is not None else 1.0
            p[22] = np.float32(float(fp) * math.log(max(_tv, 1e-6)))
            queue.write_buffer(uni, 0, p.tobytes())
            enc = dev.create_command_encoder()
            cp = enc.begin_compute_pass()
            cp.set_pipeline(pipe)
            cp.set_bind_group(0, bg)
            cp.dispatch_workgroups(int((w + 7) // 8), int((hs + 7) // 8))
            cp.end()
            _t0 = time.perf_counter()
            queue.submit([enc.finish()])
            _t_gpu += time.perf_counter() - _t0

        _t0 = time.perf_counter()
        cnt = _read(dev, queue, aCnt, bh * W, U)
        cnt_i = cnt.astype(np.int64)
        spill += int(np.maximum(cnt_i - K, 0).sum())
        _t_back += time.perf_counter() - _t0
        _t0 = time.perf_counter()
        # 归约放到 GPU：主机只读回 out/cov/neff 三个小数组（各 bh*W 个 float），
        # 不再读回 K 倍放大的桶数组，51 帧实测把 1.9s 读回 + 7s 统计降到 0.1s 级。
        n_pix = int(bh * W)
        bOut = _sbuf(dev, U, n_pix)
        bCov = _sbuf(dev, U, n_pix)
        bNef = _sbuf(dev, U, n_pix)
        ru = np.zeros(16, np.float32)
        ru[0:2].view(np.uint32)[:] = np.array([n_pix, K], np.uint32)
        ru[2] = np.float32(rej_k)
        bRU = dev.create_buffer(size=64, usage=U.UNIFORM | U.COPY_DST)
        queue.write_buffer(bRU, 0, ru.tobytes())
        rpipe = _rpipe(dev)
        rbg = dev.create_bind_group(layout=rpipe.get_bind_group_layout(0), entries=[
            {'binding': 0, 'resource': {'buffer': aVal, 'offset': 0, 'size': max(16, n_pix * K * 4)}},
            {'binding': 1, 'resource': {'buffer': aWt, 'offset': 0, 'size': max(16, n_pix * K * 4)}},
            {'binding': 2, 'resource': {'buffer': bOut, 'offset': 0, 'size': max(16, n_pix * 4)}},
            {'binding': 3, 'resource': {'buffer': bCov, 'offset': 0, 'size': max(16, n_pix * 4)}},
            {'binding': 4, 'resource': {'buffer': bNef, 'offset': 0, 'size': max(16, n_pix * 4)}},
            {'binding': 5, 'resource': {'buffer': bRU, 'offset': 0, 'size': 64}},
        ])
        enc = dev.create_command_encoder()
        cpr = enc.begin_compute_pass()
        cpr.set_pipeline(rpipe)
        cpr.set_bind_group(0, rbg)
        cpr.dispatch_workgroups(int((n_pix + 63) // 64))
        cpr.end()
        queue.submit([enc.finish()])
        o = _read(dev, queue, bOut, n_pix, U).view(np.float32)
        c = _read(dev, queue, bCov, n_pix, U).view(np.float32)
        nf = _read(dev, queue, bNef, n_pix, U).view(np.float32)
        stack[y0:y0 + bh, :] = o.reshape(bh, W)
        cov[y0:y0 + bh, :] = c.reshape(bh, W)
        neff[y0:y0 + bh, :] = nf.reshape(bh, W)
        _t_host += time.perf_counter() - _t0

    if _prof:
        print('[prof] cache %.1fs  read %.1fs  up %.1fs  gpu %.1fs  back %.1fs  host %.1fs'
              % (_t_cache, _t_read, _t_up, _t_gpu, _t_back, _t_host), flush=True)
    if log:
        log('[wgpu] 真 CFA：分桶叠加（K=%d，每像素 float 样本，判据与 CPU 一致）' % K)
        if spill:
            log('[wgpu] 桶溢出 %d 个样本（占 %.2f%%，不影响成品）'
                % (spill, 100.0 * spill / max(1, int(cov.size) * max(1, len(idx)))))
    return {'stack': stack, 'cov': cov, 'neff': neff,
            'wmax': float(cov.max()), 'rej_rate': 0.0, 'n_used': len(idx),
            'plan': plan, 'backend': 'wgpu-bucket'}
