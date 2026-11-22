# -*- coding: utf-8 -*-
"""单色传感器（OSC / CFA）支持：识别拜耳阵列、按通道拆色。

设计取舍
--------
OSC 帧是"棋盘格"：每个像素只记录 R、G、B 之一。要把它们叠成彩色，必须先把
每个像素补成三个值（debayer），再对三个平面分别做叠加与 drizzle —— 这跟 PI/WBPP
的顺序一致（校准在 CFA 域 → debayer → 配准 → 逐通道积分/drizzle）。

本模块只负责"把一帧 CFA 变成某个通道的一帧"，其余流程（配准、权重、排异、
drizzle）完全复用现有的单通道代码。这样三通道共用一次配准，省 2/3 时间。

两种拆色方式
------------
* ``bilinear``（默认）：每个通道用自身像素做加权插值补齐。绿通道像素最多、
  质量最好；红蓝各占 1/4，靠邻域补，边缘会有轻微伪彩（可用 VNG 之类改进，
  这里刻意只做双线性：快、稳、结果可预期）。
* ``superpixel``：2×2 合并成一个 RGB 单元，输出分辨率减半。颜色最干净
  （完全不插值），适合高采样/彩色只作次要信息的场景。
"""
from __future__ import annotations

import re

import os
from typing import Dict, Optional, Tuple

import numpy as np

PATTERNS = ('RGGB', 'BGGR', 'GRBG', 'GBRG')

# 每个通道要用的邻居权重（3×3 归一化卷积核），按"该像素是哪个颜色"区分


def _mask(pattern: str, ch: str) -> np.ndarray:
    """返回 2×2 内该通道的布尔掩膜（True = 这个位置是真的该通道像素）。"""
    pat = pattern.upper()
    m = np.zeros((2, 2), bool)
    # 约定：pattern 的第 0 行第 0 列 = (y%2==0, x%2==0) 的像素
    for j in range(2):
        for i in range(2):
            if pat[j * 2 + i] == ch:
                m[j, i] = True
    return m


def _tile(mask2: np.ndarray, h: int, w: int) -> np.ndarray:
    """把 2×2 掩膜铺成整幅布尔图。"""
    reps = (h + 1) // 2, (w + 1) // 2
    return np.tile(mask2, reps)[:h, :w]


_GPU = None


def _gpu_img():
    """wgpu 可分离滤波模块（懒加载；不可用时返回 False）。"""
    global _GPU
    if _GPU is None:
        try:
            if os.environ.get('DWT_CFA_GPU', '0') != '1':
                _GPU = False
            else:
                from . import gpu_wgpu_img as _m
                _GPU = _m if _m.available() else False
        except Exception:
            _GPU = False
    return _GPU


def _dilate_mean(img: np.ndarray, sel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """对 sel 选中的像素做 3×3 归一化均值填充到所有位置。

    返回 (值, 权重)：权重是参与平均的有效像素占比，用于判断边缘。

    3×3 全 1 卷积在 wgpu 上是两次可分离 box（横一遍竖一遍），
    全幅 6248×4176 单帧能从 ~2 s 降到 ~0.1 s；GPU 不可用时回落到 numpy 实现。
    """
    v = np.where(sel, img, 0.0).astype(np.float32)
    gi = _gpu_img()
    if gi:
        try:
            m3 = gi.box_blur(sel.astype(np.float32), 1)   # 3×3 均值
            num = gi.box_blur(v, 1)
            with np.errstate(invalid='ignore', divide='ignore'):
                out = np.where(m3 > 0, num / m3, 0.0)
            return out.astype(np.float32), (m3 * 9.0).astype(np.float32)
        except Exception:
            pass
    k = np.ones((3, 3), np.float32)
    num = _conv(v, k)
    den = _conv(sel.astype(np.float32), k)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(den > 0, num / den, 0.0)
    return out.astype(np.float32), den


def _conv(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    """3×3 卷积（边界用 edge 复制）。

    全 1 核走 9 次切片相加（向量化，比逐像素 Python 循环快 1~2 个数量级）；
    其它核保留通用实现。
    """
    if k.shape == (3, 3) and bool(np.all(k == 1.0)):
        # 3×3 全 1 卷积是可分离的：先横着加三列，再竖着加三行。
        # 9 次切片相加 → 6 次，实测全幅帧从 2 s 降到 0.5 s 级（结果逐位相同）。
        q = np.pad(a, ((0, 0), (1, 1)), mode='edge')
        r = q[:, :-2] + q[:, 1:-1] + q[:, 2:]
        q2 = np.pad(r, ((1, 1), (0, 0)), mode='edge')
        return q2[:-2] + q2[1:-1] + q2[2:]
    p = np.pad(a, 1, mode='edge')
    out = np.zeros_like(a, dtype=np.float32)
    h, w = a.shape
    for dy in range(3):
        for dx in range(3):
            wgt = float(k[dy, dx])
            if wgt:
                out += wgt * p[dy:dy + h, dx:dx + w]
    return out


def mask2(pattern: str, ch: str) -> np.ndarray:
    """返回该颜色在 2×2 里的掩膜（行,列）"""
    return _mask(pattern, ch)


def plane(img: np.ndarray, pattern: str, ch: str,
          y0: int = 0, x0: int = 0) -> np.ndarray:
    """从 CFA 图里取出一个通道的整幅图层（双线性思路）。

    该通道的真实像素原样保留；其余位置用 3×3 邻域内同色像素的加权均值补齐。
    """
    h, w = img.shape[:2]
    _m2 = _mask(pattern, ch)
    _ys = (np.arange(h) + int(y0)) % 2
    _xs = (np.arange(w) + int(x0)) % 2
    sel = _m2[np.ix_(_ys, _xs)]
    if sel.all():
        return img.astype(np.float32, copy=True)
    fill, _ = _dilate_mean(img, sel)
    return np.where(sel, img, fill).astype(np.float32)


def superpixel(img: np.ndarray, pattern: str) -> np.ndarray:
    """2×2 合并成 RGB，返回 (h//2, w//2, 3) float32。

    每个 2×2 内恰好有 1 红、2 绿、1 蓝（RGGB/BGGR/GRBG/GBRG 都是这个结构），
    绿取两个像素均值。
    """
    h, w = img.shape[:2]
    h2, w2 = (h // 2) * 2, (w // 2) * 2
    a = img[:h2, :w2].astype(np.float32)
    out = np.zeros((h2 // 2, w2 // 2, 3), np.float32)
    for ch, idx in (('R', 0), ('G', 1), ('B', 2)):
        m = _mask(pattern, ch)
        acc = np.zeros((h2 // 2, w2 // 2), np.float32)
        cnt = 0
        for j in range(2):
            for i in range(2):
                if m[j, i]:
                    acc += a[j::2, i::2]
                    cnt += 1
        out[:, :, idx] = acc / max(cnt, 1)
    return out


def detect_pattern(meta: Dict) -> Optional[str]:
    """从帧头里认拜耳阵列；认不出返回 None。"""
    def _clean(v) -> str:
        # 帧头里的值常带引号与补空格（PQ 写成 "'RGGB    '"），先剥掉非字母
        s = re.sub(r'[^A-Z]', '', str(v).upper())
        return s if s in PATTERNS else ''

    # 1) FITS 关键字 BAYERPAT / COLORTYP
    for kw in (meta.get('fits_keywords') or meta.get('fits_all') or []):
        name = str(kw.get('name', '')).upper()
        if name in ('BAYERPAT', 'COLORTYP', 'CFAIMAGE'):
            got = _clean(kw.get('value', ''))
            if got:
                return got
    # 2) XISF Property：PCL:CFASourcePattern
    for prop in (meta.get('props_raw') or meta.get('properties') or []):
        pid = str(prop.get('id', '')).upper()
        if 'CFASOURCEPATTERN' in pid or 'BAYERPAT' in pid:
            got = _clean(prop.get('value', ''))
            if got:
                return got
    # 3) 有些文件把 pattern 塞进 FITS 头的扁平字典
    for key in ('bayepat', 'bayerpat', 'BAYERPAT'):
        if key in meta:
            got = _clean(meta[key])
            if got:
                return got
    return None


def is_cfa(meta: Dict, img_shape: Tuple[int, ...]) -> bool:
    """判断一帧是不是未解马赛克的 CFA 帧。"""
    if len(img_shape) == 3 and img_shape[2] >= 3:
        return False
    return bool(detect_pattern(meta))
