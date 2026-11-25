"""读取 PixInsight / WBPP 的 LocalNormalization 模型（*.xnml）。

XNML 里每帧存两张网格：Scale 与 ZeroOffset（60×34 之类），像素数据是
base64 + LZ4 block + 字节 shuffle。应用方式（与 PI 输出对齐中心瓦片验证过）：

    I' = I * Scale(x, y) + ZeroOffset(x, y)

也就是「乘性比例 + 加性偏移」的逐像素仿射，把该帧的大尺度照明差拉到
参考帧的水平。网格按像素中心双线性插值到全分辨率。
"""
from __future__ import annotations

import base64
import xml.etree.ElementTree as ET
from typing import Optional, Tuple

import numpy as np

_NS = {'x': 'http://www.pixinsight.com/xnml'}


def _lz4_block(src: bytes, out_size: int) -> bytes:
    """LZ4 block 格式解压（无帧头，PI 用这个）。"""
    i = 0
    out = bytearray()
    n = len(src)
    while i < n:
        tok = src[i]
        i += 1
        lit = tok >> 4
        if lit == 15:
            while True:
                b = src[i]
                i += 1
                lit += b
                if b != 255:
                    break
        out += src[i:i + lit]
        i += lit
        if i >= n:
            break
        off = src[i] | (src[i + 1] << 8)
        i += 2
        ml = (tok & 15) + 4
        if (tok & 15) == 15:
            while True:
                b = src[i]
                i += 1
                ml += b
                if b != 255:
                    break
        s = len(out) - off
        for k in range(ml):
            out.append(out[s + k])
    return bytes(out[:out_size])


def _unshuffle(raw: bytes, w: int, h: int) -> np.ndarray:
    """PI 的 "+sh"：4 字节元素按位面重排，逆变换成 float32 网格。"""
    a = np.frombuffer(raw, np.uint8)
    if a.size != w * h * 4:
        raise ValueError(f'网格大小不符：{a.size} != {w * h * 4}')
    return a.reshape(4, -1).T.copy().view(np.float32).reshape(h, w)


def _grid(root, tag: str) -> Optional[np.ndarray]:
    el = root.find(f'.//x:{tag}', _NS)
    if el is None:
        return None
    w = int(el.get('width'))
    h = int(el.get('height'))
    sb = el.find('.//x:Subblock', _NS)
    if sb is None or not (sb.text or '').strip():
        return None
    raw = _lz4_block(base64.b64decode(sb.text.strip()),
                     int(sb.get('uncompressedSize')))
    return _unshuffle(raw, w, h)


def load_fields(path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """返回 (Scale, ZeroOffset)；没有 LN 数据时返回 None。"""
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return None
    s = _grid(root, 'Scale')
    z = _grid(root, 'ZeroOffset')
    if s is None:
        return None
    if z is None:
        z = np.zeros_like(s)
    return s.astype(np.float32), z.astype(np.float32)


def expand(grid: np.ndarray, h: int, w: int) -> np.ndarray:
    """把 (gh, gw) 网格按像素中心双线性插值到 (h, w)。"""
    gh, gw = grid.shape
    ys = (np.arange(h, dtype=np.float32) + 0.5) * (gh / float(h)) - 0.5
    xs = (np.arange(w, dtype=np.float32) + 0.5) * (gw / float(w)) - 0.5
    y0 = np.clip(np.floor(ys), 0, gh - 1).astype(np.int32)
    x0 = np.clip(np.floor(xs), 0, gw - 1).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, gh - 1)
    x1 = np.clip(x0 + 1, 0, gw - 1)
    wy = np.clip(ys - y0, 0.0, 1.0)[:, None]
    wx = np.clip(xs - x0, 0.0, 1.0)[None, :]
    g00 = grid[np.ix_(y0, x0)]
    g01 = grid[np.ix_(y0, x1)]
    g10 = grid[np.ix_(y1, x0)]
    g11 = grid[np.ix_(y1, x1)]
    top = g00 * (1 - wx) + g01 * wx
    bot = g10 * (1 - wx) + g11 * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


def apply_to(img: np.ndarray, fields) -> np.ndarray:
    """I' = I * Scale + ZeroOffset（就地，返回同一数组）。"""
    s, z = fields
    h, w = img.shape[:2]
    sf = expand(s, h, w)
    zf = expand(z, h, w)
    img *= sf
    img += zf
    return img


def sibling_path(xisf_path: str) -> str:
    """同名 .xnml（PI 默认写在源帧旁边）。"""
    p = str(xisf_path)
    for ext in ('.xisf', '.XISF', '.fits', '.FITS'):
        if p.endswith(ext):
            return p[:-len(ext)] + '.xnml'
    return p + '.xnml'
