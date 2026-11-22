"""把已有的三个单通道成品重新合成一张彩色 XISF（不做白平衡）。

用途：单通道已经叠好（甚至各自在 PI 里处理过）之后，想重新合一次色；
或者想用不同的通道重新组合。合成只做三件事：

1. 几何对齐检查（三张必须同尺寸；不同则报错退出，不猜）
2. 可选 `--sky-match`：把 R/B 的天空中位数拉到 G 的水平（默认关）
3. 写 XISF（H×W×3、Normal 存储、bounds 按数据实际范围）

**不做白平衡/色彩校正**——按用户要求，通道比例保持原样。

用法：
    python recompose_rgb.py --r PWS_x_R.xisf --g PWS_x_G.xisf --b PWS_x_B.xisf \
        --out PWS_x_RGB.xisf [--sky-match] [--preview 预览.png]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np


def _load(path: str) -> np.ndarray:
    from xisf_io import read_xisf
    with contextlib.redirect_stdout(io.StringIO()):
        im = read_xisf(str(path))[0]
    d = np.asarray(im.data, dtype=np.float32)
    if d.ndim == 3:
        d = d[:, :, 0]
    return d


def _sky(a: np.ndarray) -> float:
    v = a[np.isfinite(a)]
    return float(np.median(v)) if v.size else float('nan')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='三通道重新合成彩色 XISF（不做白平衡）')
    ap.add_argument('--r', required=True)
    ap.add_argument('--g', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sky-match', action='store_true',
                    help='把 R/B 的天空中位数拉到 G 的水平（默认关，保持原始比例）')
    ap.add_argument('--preview', default='', help='可选：同时输出一张 PNG 预览')
    a = ap.parse_args(argv)

    chans = []
    for name, p in (('R', a.r), ('G', a.g), ('B', a.b)):
        if not Path(p).exists():
            print(f'[错误] {name} 通道文件不存在：{p}')
            return 2
        chans.append(_load(p))
    shapes = {c.shape for c in chans}
    if len(shapes) != 1:
        print('[错误] 三张尺寸不一致：%s' % ', '.join(str(c.shape) for c in chans))
        print('       请先用同一画布重叠，或在 PI 里裁齐后再合成。')
        return 2

    R, G, B = chans
    if a.sky_match:
        sg = _sky(G)
        for name, arr in (('R', R), ('B', B)):
            s = _sky(arr)
            if np.isfinite(s) and np.isfinite(sg) and s > 0:
                k = sg / s
                arr *= np.float32(k)
                print('[天空] %s 通道 ×%.5f（%.4e → %.4e）' % (name, k, s, s * k))
    else:
        print('[天空] 未做通道平衡（保持原始比例）')

    rgb = np.stack([R, G, B], axis=-1)
    finite = np.isfinite(rgb)
    lo = float(np.nanmin(rgb)) if finite.any() else 0.0
    hi = float(np.nanmax(rgb)) if finite.any() else 1.0
    if not (hi > lo):
        hi = lo + 1.0

    from xisf_io import XISFImage, write_xisf
    h, w, _ = rgb.shape
    img = XISFImage()
    img.data = np.ascontiguousarray(rgb, dtype=np.float32)
    img.sample_format = 'Float32'
    img.geometry = (w, h, 3)
    img.color_space = 'RGB'
    img.pixel_storage = 'Normal'
    img.bounds = (lo, hi)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_xisf(str(out), img)
    print('[输出] %s  %d×%d×3  bounds=(%.4g, %.4g)' % (out, w, h, lo, hi))

    if a.preview:
        try:
            # 用包内自带的 PNG 编码器（maps_page._to_png），不依赖 Pillow
            from maps_page import _to_png
            # 预览用「中位 ± MAD」定标（和成品预览一致）；边界值会被饱和星点撑到 1，
            # 按 bounds 拉伸整幅会几乎全黑。
            _lo = np.zeros(3, np.float32); _hi = np.zeros(3, np.float32)
            for _c in range(3):
                _v = rgb[..., _c]
                _m = float(np.median(_v))
                _d = float(np.median(np.abs(_v - _m))) * 1.4826
                _lo[_c] = max(0.0, _m - 2.8 * _d)
                _hi[_c] = _m + 60.0 * _d
            x = np.clip((rgb - _lo) / np.maximum(_hi - _lo, 1e-12), 0, 1)
            x = np.arcsinh(x * 18.0) / np.arcsinh(18.0)
            u8 = (x * 255).astype(np.uint8)
            sg = max(1, int(np.ceil(max(h, w) / 1200.0)))
            u8 = u8[::sg, ::sg]
            Path(a.preview).write_bytes(_to_png(u8))
            print('[预览] %s  %d×%d（抽样 %d）' % (a.preview, u8.shape[1], u8.shape[0], sg))
        except Exception as e:
            print('[预览] 生成失败（不影响 XISF）：%s' % e)
    return 0


if __name__ == '__main__':
    sys.exit(main())
