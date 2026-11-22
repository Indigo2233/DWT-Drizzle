"""R 场的可分离滤波搬到 wgpu（box / max / gauss）。

R 场本身是平滑的支撑掩膜，真正贵的是三个邻域原语（全幅 6248×4176 时各约 1~3 s，
叠加起来占 `build_r_field` 的大头）。它们都是可分离的：先横一遍再竖一遍。
这里用同一个 WGSL 内核（kind 0=box、1=max、2=gauss）跑两遍，省掉 scipy 的
内存来回拷贝，也让 GPU 在配准之后、叠加之前就有活干。

对外只暴露三个函数，入参出参都是 float32 ndarray、形状不变；
任何异常都抛给调用方，由 `core/pws.py` 决定是否回退到 scipy。
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_GPUD = Path(__file__).resolve().parent.parent / 'gpudeps'
if _GPUD.is_dir() and str(_GPUD) not in sys.path:
    sys.path.insert(0, str(_GPUD))

_WGSL = """
struct Q { w: u32, h: u32, r: u32, kind: u32, vert: u32, p0: u32, p1: u32, p2: u32 };

@group(0) @binding(0) var<uniform> Q_: Q;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read> wt: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = Q_.w * Q_.h;
  let i = gid.x;
  if (i >= n) { return; }
  let y = i / Q_.w;
  let x = i % Q_.w;
  let r = i32(Q_.r);
  var s = 0.0;
  var m = src[i];
  let horiz = Q_.vert == 0u;
  for (var k = -r; k <= r; k = k + 1) {
    var yy = i32(y);
    var xx = i32(x);
    if (horiz) { xx = clamp(xx + k, 0, i32(Q_.w) - 1); }
    else { yy = clamp(yy + k, 0, i32(Q_.h) - 1); }
    let v = src[u32(yy) * Q_.w + u32(xx)];
    if (Q_.kind == 0u) { s = s + v; }
    else if (Q_.kind == 1u) { m = max(m, v); }
    else { s = s + v * wt[u32(k + r)]; }
  }
  if (Q_.kind == 0u) { dst[i] = s / f32(2 * r + 1); }
  else if (Q_.kind == 1u) { dst[i] = m; }
  else { dst[i] = s; }
}
"""

_cache = {}


def _dev():
    if 'dev' in _cache:
        return _cache['dev'], _cache['queue'], _cache['pipe'], _cache['bgl']
    import wgpu
    ad = wgpu.gpu.request_adapter_sync(power_preference='high-performance')
    dev = ad.request_device_sync()
    sm = dev.create_shader_module(code=_WGSL)
    bgl = dev.create_bind_group_layout(entries=[
        {'binding': 0, 'visibility': wgpu.ShaderStage.COMPUTE,
         'buffer': {'type': wgpu.BufferBindingType.uniform}},
        {'binding': 1, 'visibility': wgpu.ShaderStage.COMPUTE,
         'buffer': {'type': wgpu.BufferBindingType.read_only_storage}},
        {'binding': 2, 'visibility': wgpu.ShaderStage.COMPUTE,
         'buffer': {'type': wgpu.BufferBindingType.storage}},
        {'binding': 3, 'visibility': wgpu.ShaderStage.COMPUTE,
         'buffer': {'type': wgpu.BufferBindingType.read_only_storage}},
    ])
    pipe = dev.create_compute_pipeline(layout=dev.create_pipeline_layout(bind_group_layouts=[bgl]),
                                       compute={'module': sm, 'entry_point': 'main'})
    _cache.update(dev=dev, queue=dev.queue, pipe=pipe, bgl=bgl)
    return dev, dev.queue, pipe, bgl


def _run(a: np.ndarray, radius: int, kind: int, vert: bool,
         wt: Optional[np.ndarray] = None) -> np.ndarray:
    import wgpu
    a = np.ascontiguousarray(a, np.float32)
    h, w = a.shape
    n = h * w
    dev, queue, pipe, bgl = _dev()
    U = wgpu.BufferUsage

    def sbuf(size):
        return dev.create_buffer(size=max(16, int(size)), usage=U.STORAGE | U.COPY_SRC | U.COPY_DST)

    bs = sbuf(n * 4); queue.write_buffer(bs, 0, memoryview(a))
    bd = sbuf(n * 4)
    r = int(radius)
    wk = wt if wt is not None else np.zeros(1, np.float32)
    bw = sbuf(max(16, len(wk) * 4)); queue.write_buffer(bw, 0, memoryview(np.ascontiguousarray(wk, np.float32)))
    q = np.zeros(8, np.uint32)
    q[0], q[1], q[2], q[3] = w, h, r, kind
    q[4] = 1 if vert else 0
    bu = dev.create_buffer(size=32, usage=U.UNIFORM | U.COPY_DST)
    queue.write_buffer(bu, 0, memoryview(q))

    bg = dev.create_bind_group(layout=bgl, entries=[
        {'binding': 0, 'resource': {'buffer': bu, 'offset': 0, 'size': 32}},
        {'binding': 1, 'resource': {'buffer': bs, 'offset': 0, 'size': n * 4}},
        {'binding': 2, 'resource': {'buffer': bd, 'offset': 0, 'size': n * 4}},
        {'binding': 3, 'resource': {'buffer': bw, 'offset': 0, 'size': max(16, len(wk) * 4)}},
    ])
    enc = dev.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(pipe)
    cp.set_bind_group(0, bg)
    cp.dispatch_workgroups(max(1, (n + 63) // 64))
    cp.end()
    queue.submit([enc.finish()])
    out = np.frombuffer(dev.queue.read_buffer(bd), np.float32, count=n).reshape(h, w)
    return np.array(out, np.float32, copy=True)


def box_blur(a: np.ndarray, radius: int) -> np.ndarray:
    return _run(_run(a, radius, 0, False), radius, 0, True)


def max_filter(a: np.ndarray, radius: int) -> np.ndarray:
    return _run(_run(a, radius, 1, False), radius, 1, True)


def gauss_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    sig = float(max(1e-3, sigma))
    r = int(max(1, math.ceil(3.0 * sig)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sig) ** 2)
    k /= k.sum()
    return _run(_run(a, r, 2, False, k.astype(np.float32)),
                r, 2, True, k.astype(np.float32))


def available() -> bool:
    try:
        import wgpu  # noqa: F401
        _dev()
        return True
    except Exception:
        return False
