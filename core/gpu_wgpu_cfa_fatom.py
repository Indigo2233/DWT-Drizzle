"""真·散射累加后端（float32 原子加）。

与分桶后端的区别：不保存样本，直接对输出像素做原子累加，显存只占 6 个画布大小的
float 缓冲；每条带把帧数据上传一次，三遍扫描都在显存里完成。

数学与 CPU 的 ``_scatter_band`` 一致：每个源像素按液滴盒与输出像素的重叠面积
``wx*wy/P^2`` 摊到相邻像素，权重 ``w = C^(A*R) * S^(B*(1-R)) * T^fp``。

三遍：
  pass0  Σw, Σwv, Σw²      → μ
  pass1  Σw·min(|v−μ|,3σ)  → winsor 尺度 s = 1.15·Σw|dev|/Σw
  pass2  只保留 |v−μ| ≤ k·s → Σw, Σwv, Σw²（成品）
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np

try:  # 既支持 -m core.pws，也支持直接跑脚本
    from . import gpu_wgpu as G
except ImportError:  # pragma: no cover
    import gpu_wgpu as G


_SCATTER = """
struct QP {
  a: f32, b: f32, c: f32, d: f32, e: f32, f: f32,
  w_src: u32, h_src: u32, w_out: u32, h_out: u32,
  y_off: u32, c0: u32, r0: u32, mask: u32,
  P: f32, lnc: f32, lns: f32, wa: f32, wb: f32, has_r: f32, kclip: f32,
  mode: f32, s_off: u32, lnt: f32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> rmap: array<f32>;
@group(0) @binding(2) var<uniform> Q: QP;
@group(0) @binding(3) var<storage, read_write> aW: array<atomic<f32>>;
@group(0) @binding(4) var<storage, read_write> aV: array<atomic<f32>>;
@group(0) @binding(5) var<storage, read_write> aDev: array<atomic<f32>>;
@group(0) @binding(6) var<storage, read> mu: array<f32>;
@group(0) @binding(7) var<storage, read> sd: array<f32>;
@group(0) @binding(8) var<storage, read_write> aW2: array<atomic<f32>>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let px = gid.x;
  let py = gid.y;
  if (px >= Q.w_src || py >= Q.h_src) { return; }
  let gx0 = px + Q.c0;
  let gy0 = py + Q.r0;
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
  let mode = i32(Q.mode);

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
      let rr = select(0.5, rmap[(u32(gy) + Q.y_off) * Q.w_out + u32(kx)], Q.has_r > 0.5);
      let wgt = exp(Q.wa * rr * Q.lnc + Q.wb * (1.0 - rr) * Q.lns);
      let w = wgt * wx * wy / (Q.P * Q.P) * exp(Q.lnt);

      if (mode == 0) {
        atomicAdd(&aW[idx], w);
        atomicAdd(&aV[idx], w * v);
        atomicAdd(&aW2[idx], w * w);
      } else if (mode == 1) {
        atomicAdd(&aDev[idx], w * abs(v - mu[idx]));
      } else if (mode == 4) {
        let m = mu[idx];
        let lim = sd[idx];
        atomicAdd(&aV[idx], w * clamp(v, m - lim, m + lim));
      } else if (mode == 3) {
        // winsor 第二段：截尾到 k·σ（lim 存在 sd 里）
        atomicAdd(&aDev[idx], w * min(abs(v - mu[idx]), sd[idx]));
      } else {
        // 分块尺度只在抽样点上有值；其余像素退回本像素的 1.15·Σw|dev|/Σw
        let sref = select(1.15 * aDev[idx] / max(aW[idx], 1e-20), sd[idx], sd[idx] > 1e-20);
        if (Q.kclip > 0.0 && abs(v - mu[idx]) > Q.kclip * sref) { continue; }
        atomicAdd(&aW[idx], w);
        atomicAdd(&aV[idx], w * v);
        atomicAdd(&aW2[idx], w * w);
      }
    }
  }
}
"""

_STATS = """
struct S { n: u32, mode: u32, w_out: u32, h_out: u32, blk: u32, st: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> aW: array<f32>;
@group(0) @binding(1) var<storage, read> aV: array<f32>;
@group(0) @binding(2) var<storage, read> aD: array<f32>;
@group(0) @binding(3) var<storage, read_write> mu: array<f32>;
@group(0) @binding(4) var<storage, read_write> sd: array<f32>;
@group(0) @binding(5) var<uniform> U: S;

// 第一遍：逐像素 μ
@compute @workgroup_size(64)
fn mean_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= U.n) { return; }
  let w = aW[i];
  if (w > 1e-20) { mu[i] = aV[i] / w; }
}

// 第二遍：与 CPU 同口径的分块 winsor 尺度（blk=64、st=3 抽样、1.15×）
@compute @workgroup_size(64)
fn block_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
  // 每线程负责一个 st×st 抽样点，归约由主机侧完成（这里只写该点的 |dev|/w 到 sd）
  let i = gid.x;
  if (i >= U.n) { return; }
  let x = i % U.w_out;
  let y = i / U.w_out;
  if ((x % U.st) != 0u || (y % U.st) != 0u) { return; }
  let w = aW[i];
  if (w <= 1e-20) { return; }
  sd[i] = aD[i] / w;
}
"""

_FINAL = """
struct F { n: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> aW: array<f32>;
@group(0) @binding(1) var<storage, read> aV: array<f32>;
@group(0) @binding(2) var<storage, read> aW2: array<f32>;
@group(0) @binding(3) var<storage, read_write> oOut: array<f32>;
@group(0) @binding(4) var<storage, read_write> oCov: array<f32>;
@group(0) @binding(5) var<storage, read_write> oNef: array<f32>;
@group(0) @binding(6) var<uniform> U: F;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= U.n) { return; }
  let w = aW[i];
  if (w <= 1e-20) { oOut[i] = 0.0; oCov[i] = 0.0; oNef[i] = 0.0; return; }
  oOut[i] = aV[i] / w;
  oCov[i] = w;
  oNef[i] = (w * w) / max(aW2[i], 1e-20);
}
"""

_PARITY = {
    'RGGB': {'R': 0b0001, 'G': 0b0110, 'B': 0b1000},
    'BGGR': {'R': 0b1000, 'G': 0b0110, 'B': 0b0001},
    'GRBG': {'R': 0b0010, 'G': 0b1001, 'B': 0b0100},
    'GBRG': {'R': 0b0100, 'G': 0b1001, 'B': 0b0010},
}

_CACHE: Dict[str, object] = {}


def available() -> bool:
    try:
        return bool(G.available() and G.has_f32_atomic())
    except Exception:
        return False


def _pipes(dev):
    if 'p' in _CACHE:
        return _CACHE['p'], _CACHE['ps'], _CACHE['pf'], _CACHE['pbs']
    mp = dev.create_shader_module(code=_SCATTER)
    ms = dev.create_shader_module(code=_STATS)
    mf = dev.create_shader_module(code=_FINAL)
    p = dev.create_compute_pipeline(layout='auto', compute={'module': mp, 'entry_point': 'main'})
    ps = dev.create_compute_pipeline(layout='auto', compute={'module': ms, 'entry_point': 'mean_pass'})
    pbs = dev.create_compute_pipeline(layout='auto', compute={'module': ms, 'entry_point': 'block_scale'})
    pf = dev.create_compute_pipeline(layout='auto', compute={'module': mf, 'entry_point': 'main'})
    _CACHE.update({'p': p, 'ps': ps, 'pf': pf, 'pbs': pbs})
    return p, ps, pf, pbs


def stack_cfa_fatom(fs, C, S, dx, dy, th, kk, plan, A, B, rej_k, rej_gate,
                    R=None, pixfrac=1.0, band_rows=512, T=None, fp=1.0,
                    log=None, pattern=None, channel=None, **kw):
    if not available():
        raise RuntimeError('适配器不支持 shader-float32-atomic')
    dev, queue = G._device_f32()
    wgpu = G._wp()
    U = wgpu.BufferUsage

    H, W = int(plan['H']), int(plan['W'])
    s_grid = float(plan.get('s', 1.0))
    ox, oy = float(plan.get('ox', 0.0)), float(plan.get('oy', 0.0))
    h, w = int(fs.shape[0]), int(fs.shape[1])
    P = max(1e-6, float(pixfrac) * s_grid)

    n = len(dx)
    idx = [i for i in range(n) if np.isfinite(dx[i]) and np.isfinite(dy[i])
           and (th is None or np.isfinite(th[i]))]
    if len(idx) < 2:
        raise RuntimeError('有效帧不足')

    mask2 = _PARITY[pattern][channel] if pattern else 0b1111
    stack = np.zeros((H, W), np.float32)
    cov = np.zeros((H, W), np.float32)
    neff = np.zeros((H, W), np.float32)

    p = np.zeros(27, np.float32)
    pu = p.view(np.uint32)
    pu[13] = np.uint32(mask2)
    p[14] = np.float32(P)
    p[17] = np.float32(float(A))
    p[18] = np.float32(float(B))
    p[19] = np.float32(1.0 if R is not None else 0.0)

    pipe, pstat, pfin, pbs = _pipes(dev)
    bgl = pipe.get_bind_group_layout(0)

    def sbuf(nel):
        return dev.create_buffer(size=max(16, int(nel) * 4),
                                 usage=U.STORAGE | U.COPY_SRC | U.COPY_DST)

    for y0 in range(0, H, band_rows):
        bh = min(band_rows, H - y0)
        n_pix = int(bh * W)
        aW, aV, aD, aW2 = sbuf(n_pix), sbuf(n_pix), sbuf(n_pix), sbuf(n_pix)
        bMu, bSd = sbuf(n_pix), sbuf(n_pix)
        oOut, oCov, oNef = sbuf(n_pix), sbuf(n_pix), sbuf(n_pix)

        # 统一的元数据 uniform：w_src/h_src/w_out/h_out/y_off/r0
        pu[6] = np.uint32(w)
        pu[8] = np.uint32(W)
        pu[9] = np.uint32(bh)
        pu[10] = np.uint32(y0)
        pu[11] = np.uint32(0)
        bQ = dev.create_buffer(size=128, usage=U.UNIFORM | U.COPY_DST)

        def zero_acc():
            for b in (aW, aV, aD, aW2):
                queue.write_buffer(b, 0, np.zeros(n_pix, np.float32).tobytes())

        zero_acc()
        for y in range(bh):
            pass

        # 源行窗口（逐帧逆变换取并集；旋转接近 180° 时源行方向相反）
        lo, hi = 1 << 30, -1
        for j in idx:
            th_i = 0.0 if th is None else float(th[j])
            ct, stn = float(np.cos(th_i)), float(np.sin(th_i))
            ks = float(kk[j])
            d_ = s_grid * ks * stn
            e_ = s_grid * ks * ct
            f_ = s_grid * -float(dy[j]) + oy
            for gx in (0, w - 1):
                for Yb in (y0 - 2 * P - 4, y0 + bh + 2 * P + 4):
                    if abs(e_) < 1e-9:
                        continue
                    gy = (Yb - d_ * (gx + 0.5) - f_) / e_ - 0.5
                    if np.isfinite(gy):
                        lo = min(lo, int(np.floor(gy)))
                        hi = max(hi, int(np.ceil(gy)))
        if os.environ.get('DWT_FATOM_FULLSTRIP', '').strip() == '1':
            lo, hi = 0, h
        lo = max(0, lo - 8)
        hi = min(h, hi + 8)
        if hi <= lo:
            continue
        hs = hi - lo
        plane = hs * w
        pu[7] = np.uint32(hs)
        pu[12] = np.uint32(lo)

        buf = np.empty((len(idx), plane), np.float32)
        for k, j in enumerate(idx):
            buf[k] = np.asarray(fs.frame(int(j)), np.float32)[lo:hi, :].ravel()
        bSrc = sbuf(len(idx) * plane)
        queue.write_buffer(bSrc, 0, buf.tobytes())
        del buf

        if R is not None:
            rr = np.asarray(R, np.float32)
            if rr.shape != (H, W):
                yy = (np.arange(H) * rr.shape[0] // H).clip(0, rr.shape[0] - 1)
                xx = (np.arange(W) * rr.shape[1] // W).clip(0, rr.shape[1] - 1)
                rr = rr[np.ix_(yy, xx)]
            bR = sbuf(H * W)
            queue.write_buffer(bR, 0, np.ascontiguousarray(rr).tobytes())
            rsz = H * W * 4
        else:
            bR = sbuf(4)
            queue.write_buffer(bR, 0, np.zeros(4, np.float32).tobytes())
            rsz = 16

        def scatter(mode, kclip):
            p[20] = np.float32(kclip)
            p[21] = np.float32(float(mode))
            enc = dev.create_command_encoder()
            cp = enc.begin_compute_pass()
            for k, j in enumerate(idx):
                th_i = 0.0 if th is None else float(th[j])
                ct, stn = float(np.cos(th_i)), float(np.sin(th_i))
                ks = float(kk[j])
                p[0] = np.float32(s_grid * ks * ct)
                p[1] = np.float32(-s_grid * ks * stn)
                p[2] = np.float32(s_grid * -float(dx[j]) + ox)
                p[3] = np.float32(s_grid * ks * stn)
                p[4] = np.float32(s_grid * ks * ct)
                p[5] = np.float32(s_grid * -float(dy[j]) + oy)
                pu[22] = np.uint32(k * plane)
                if C is not None:
                    p[15] = np.float32(np.log(max(float(C[int(j)]), 1e-6)))
                if S is not None:
                    p[16] = np.float32(np.log(max(float(S[int(j)]), 1e-6)))
                tv = float(T[int(j)]) if T is not None else 1.0
                p[23] = np.float32(float(fp) * float(np.log(max(tv, 1e-6))))
                bQk = dev.create_buffer(size=128, usage=U.UNIFORM | U.COPY_DST)
                queue.write_buffer(bQk, 0, p.tobytes())
                if os.environ.get('DWT_FATOM_DBG') == '1' and k == 0 and y0 == 0:
                    print('[fatom-dbg] frame %d a=%.6f b=%.6f c=%.4f d=%.6f e=%.6f f=%.4f lo=%d hs=%d'
                          % (j, p[0], p[1], p[2], p[3], p[4], p[5], lo, hs), flush=True)
                    print('[fatom-dbg] X(center)=%.4f Y(center)=%.4f' % (
                        p[0]*(w*0.5+0.5) + p[1]*(hs*0.5+0.5) + p[2],
                        p[3]*(w*0.5+0.5) + p[4]*(hs*0.5+0.5) + p[5]), flush=True)
                bg = dev.create_bind_group(layout=bgl, entries=[
                    {'binding': 0, 'resource': {'buffer': bSrc, 'offset': 0, 'size': len(idx) * plane * 4}},
                    {'binding': 1, 'resource': {'buffer': bR, 'offset': 0, 'size': rsz}},
                    {'binding': 2, 'resource': {'buffer': bQk, 'offset': 0, 'size': 128}},
                    {'binding': 3, 'resource': {'buffer': aW, 'offset': 0, 'size': n_pix * 4}},
                    {'binding': 4, 'resource': {'buffer': aV, 'offset': 0, 'size': n_pix * 4}},
                    {'binding': 5, 'resource': {'buffer': aD, 'offset': 0, 'size': n_pix * 4}},
                    {'binding': 6, 'resource': {'buffer': bMu, 'offset': 0, 'size': n_pix * 4}},
                    {'binding': 7, 'resource': {'buffer': bSd, 'offset': 0, 'size': n_pix * 4}},
                    {'binding': 8, 'resource': {'buffer': aW2, 'offset': 0, 'size': n_pix * 4}},
                ])
                cp.set_pipeline(pipe)
                cp.set_bind_group(0, bg)
                cp.dispatch_workgroups(int((w + 7) // 8), int((hs + 7) // 8))
            cp.end()
            queue.submit([enc.finish()])

        def _u32_uni(vals):
            u = np.zeros(8, np.uint32)
            for i, v in enumerate(vals):
                u[i] = np.uint32(v)
            b = dev.create_buffer(size=32, usage=U.UNIFORM | U.COPY_DST)
            queue.write_buffer(b, 0, u.tobytes())
            return b

        def mean_pass():
            bU = _u32_uni([n_pix, 0, W, bh, 64, 3])
            # mean_pass 只用到 aW(0)/aV(1)/mu(3)/U(5)，`layout='auto'` 生成的布局只含这 4 个绑定
            bg = dev.create_bind_group(layout=pstat.get_bind_group_layout(0), entries=[
                {'binding': 0, 'resource': {'buffer': aW, 'offset': 0, 'size': n_pix * 4}},
                {'binding': 1, 'resource': {'buffer': aV, 'offset': 0, 'size': n_pix * 4}},
                {'binding': 3, 'resource': {'buffer': bMu, 'offset': 0, 'size': n_pix * 4}},
                {'binding': 5, 'resource': {'buffer': bU, 'offset': 0, 'size': 32}},
            ])
            enc = dev.create_command_encoder()
            cp = enc.begin_compute_pass()
            cp.set_pipeline(pstat)
            cp.set_bind_group(0, bg)
            cp.dispatch_workgroups(int((n_pix + 63) // 64))
            cp.end()
            queue.submit([enc.finish()])

        def block_scale(k_clip=1.0, scale=1.15):
            """与 CPU 的 _winsor_scale 同口径：64×64 块、每 3 像素取 1、1.15×平均 |dev|。"""
            bU = _u32_uni([n_pix, 1, W, bh, 64, 3])
            # block_scale 只用到 aW(0)/aD(2)/sd(4)/U(5)
            bg = dev.create_bind_group(layout=pbs.get_bind_group_layout(0), entries=[
                {'binding': 0, 'resource': {'buffer': aW, 'offset': 0, 'size': n_pix * 4}},
                {'binding': 2, 'resource': {'buffer': aD, 'offset': 0, 'size': n_pix * 4}},
                {'binding': 4, 'resource': {'buffer': bSd, 'offset': 0, 'size': n_pix * 4}},
                {'binding': 5, 'resource': {'buffer': bU, 'offset': 0, 'size': 32}},
            ])
            enc = dev.create_command_encoder()
            cp = enc.begin_compute_pass()
            cp.set_pipeline(pbs)
            cp.set_bind_group(0, bg)
            cp.dispatch_workgroups(int((n_pix + 63) // 64))
            cp.end()
            queue.submit([enc.finish()])
            # 抽样点已在 sd 里，主机侧按块求均值并展开
            sp = np.frombuffer(queue.read_buffer(bSd), np.float32)[:n_pix].reshape(bh, W)
            blk = 64
            out = np.zeros((bh, W), np.float32)
            for by in range(0, bh, blk):
                for bx in range(0, W, blk):
                    sub = sp[by:by + blk:3, bx:bx + blk:3]
                    vals = sub[sub > 0]
                    s_blk = float(np.mean(vals)) * float(scale) if vals.size else 0.0
                    out[by:by + blk, bx:bx + blk] = s_blk
            queue.write_buffer(bSd, 0, out.tobytes())

        scatter(0, 0.0)   # Σw Σwv Σw²
        mean_pass()       # μ
        scatter(1, 0.0)                     # Σw|v−μ0|
        block_scale(k_clip=3.0, scale=1.15)  # 粗尺度 s0（写回 sd，作为 mode4/3 的截尾半径）
        # ---- 稳健锚点：截尾均值，接近 CPU 的逐像素中位数 ----
        queue.write_buffer(aV, 0, np.zeros(n_pix, np.float32).tobytes())
        scatter(4, 0.0)                     # Σw·clip(v, μ0±3s0)
        wsum = np.frombuffer(queue.read_buffer(aW), np.float32)[:n_pix]
        vsum = np.frombuffer(queue.read_buffer(aV), np.float32)[:n_pix]
        mu_r = np.where(wsum > 1e-20, vsum / np.maximum(wsum, 1e-20), 0.0).astype(np.float32)
        queue.write_buffer(bMu, 0, mu_r.tobytes())
        queue.write_buffer(aD, 0, np.zeros(n_pix, np.float32).tobytes())
        scatter(1, 0.0)                     # Σw|v−μ_robust|
        block_scale(k_clip=1.0, scale=1.15)  # 最终 winsor 尺度

        queue.write_buffer(aW, 0, np.zeros(n_pix, np.float32).tobytes())
        queue.write_buffer(aV, 0, np.zeros(n_pix, np.float32).tobytes())
        queue.write_buffer(aW2, 0, np.zeros(n_pix, np.float32).tobytes())
        scatter(2, float(rej_k))

        u = np.zeros(4, np.float32)
        u.view(np.uint32)[0] = np.uint32(n_pix)
        bU = dev.create_buffer(size=16, usage=U.UNIFORM | U.COPY_DST)
        queue.write_buffer(bU, 0, u.tobytes())
        bg = dev.create_bind_group(layout=pfin.get_bind_group_layout(0), entries=[
            {'binding': 0, 'resource': {'buffer': aW, 'offset': 0, 'size': n_pix * 4}},
            {'binding': 1, 'resource': {'buffer': aV, 'offset': 0, 'size': n_pix * 4}},
            {'binding': 2, 'resource': {'buffer': aW2, 'offset': 0, 'size': n_pix * 4}},
            {'binding': 3, 'resource': {'buffer': oOut, 'offset': 0, 'size': n_pix * 4}},
            {'binding': 4, 'resource': {'buffer': oCov, 'offset': 0, 'size': n_pix * 4}},
            {'binding': 5, 'resource': {'buffer': oNef, 'offset': 0, 'size': n_pix * 4}},
            {'binding': 6, 'resource': {'buffer': bU, 'offset': 0, 'size': 16}},
        ])
        enc = dev.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pfin)
        cp.set_bind_group(0, bg)
        cp.dispatch_workgroups(int((n_pix + 63) // 64))
        cp.end()
        queue.submit([enc.finish()])

        def rd(b):
            return np.frombuffer(queue.read_buffer(b), np.float32).copy()

        stack[y0:y0 + bh, :] = rd(oOut).reshape(bh, W)
        cov[y0:y0 + bh, :] = rd(oCov).reshape(bh, W)
        neff[y0:y0 + bh, :] = rd(oNef).reshape(bh, W)

    if log:
        log('[wgpu] 真 CFA：散射累加（float 原子，三遍）')
    return {'stack': stack, 'cov': cov, 'neff': neff, 'rej_rate': 0.0,
            'wmax': float(cov.max()) if cov.size else 0.0}
