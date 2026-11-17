# -*- coding: utf-8 -*-
"""自动裁掉没有覆盖的边缘（drizzle 画布四角与只被少数帧覆盖的带子）。

输入是「每像素被多少帧覆盖」的计数图（`stack_drizzle` 的 `ncov`）。流程：
按 frac × 最大帧数 取掩膜 → 9×9 闭运算填掉落点造成的针孔 → 粗网格求最大
内接矩形 → 回原分辨率逐边外扩。闭运算这一步不能省：不填针孔，最大矩形会
被一粒粒麻点切成几百像素。
"""
import numpy as np


def _max_rect(mask):
    """最大内接矩形 → (y0, y1, x0, x1)，左闭右开；找不到返回 None。"""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or not m.any():
        return None
    h, w = m.shape
    heights = np.zeros(w, dtype=np.int32)
    best_area = -1
    best = None
    for y in range(h):
        heights = np.where(m[y], heights + 1, 0)
        stack = []
        for x in range(w + 1):
            cur = int(heights[x]) if x < w else 0
            start = x
            while stack and stack[-1][1] >= cur:
                sx, sh = stack.pop()
                area = sh * (x - sx)
                if area > best_area:
                    best_area = area
                    best = (y - sh + 1, y + 1, sx, x)
                start = sx
            stack.append((start, cur))
    if best is None or best_area <= 0:
        return None
    y0, y1, x0, x1 = best
    return int(y0), int(y1), int(x0), int(x1)


def _ok(intsum, y0, y1, x0, x1):
    s = int(intsum[y1, x1] - intsum[y0, x1] - intsum[y1, x0] + intsum[y0, x0])
    return s == (y1 - y0) * (x1 - x0)


def _refine(intsum, box, h, w, step=16):
    y0, y1, x0, x1 = box
    moved = True
    while moved:
        moved = False
        while y0 - step >= 0 and _ok(intsum, y0 - step, y1, x0, x1):
            y0 -= step
            moved = True
        while y1 + step <= h and _ok(intsum, y0, y1 + step, x0, x1):
            y1 += step
            moved = True
        while x0 - step >= 0 and _ok(intsum, y0, y1, x0 - step, x1):
            x0 -= step
            moved = True
        while x1 + step <= w and _ok(intsum, y0, y1, x0, x1 + step):
            x1 += step
            moved = True
    for _ in range(4):
        while y0 > 0 and _ok(intsum, y0 - 1, y1, x0, x1):
            y0 -= 1
        while y1 < h and _ok(intsum, y0, y1 + 1, x0, x1):
            y1 += 1
        while x0 > 0 and _ok(intsum, y0, y1, x0 - 1, x1):
            x0 -= 1
        while x1 < w and _ok(intsum, y0, y1, x0, x1 + 1):
            x1 += 1
    return int(y0), int(y1), int(x0), int(x1)


def max_rect(mask, block=8, keep=0.99):
    """掩膜上的最大内接矩形（粗网格投票 + 原分辨率精修）。"""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or not m.any():
        return None
    h, w = m.shape
    if block > 1 and min(h, w) > 4 * block:
        gh, gw = h // block, w // block
        blk = m[:gh * block, :gw * block].reshape(gh, block, gw, block)
        small = blk.mean(axis=(1, 3)) >= float(keep)
        sb = _max_rect(small)
        if sb is None:
            return None
        box = (sb[0] * block, sb[1] * block, sb[2] * block, sb[3] * block)
        intsum = np.zeros((h + 1, w + 1), np.int64)
        intsum[1:, 1:] = np.cumsum(np.cumsum(m, 0), 1)
        return _refine(intsum, box, h, w)
    return _max_rect(m)


def crop_box(ncov, frac=0.9, min_frames=3.0, close=9):
    """按「覆盖帧数 ≥ max(min_frames, frac × 最大帧数)」取最大内接矩形。"""
    a = np.asarray(ncov, np.float64)
    if a.ndim != 2 or not a.size:
        return None
    top = float(np.nanmax(a))
    if not np.isfinite(top) or top <= 0:
        return None
    thr = max(float(min_frames), float(frac) * top)
    mask = np.isfinite(a) & (a >= thr)
    if not mask.any():
        return None
    if close >= 3:
        try:
            from scipy import ndimage
            # border_value=1：否则闭运算会把贴着画幅边的 4 像素当成外面而啃掉
            mask = ndimage.binary_closing(mask, np.ones((int(close), int(close)), bool),
                                          border_value=1)
        except Exception:
            pass
    return max_rect(mask)

def noise_map(img, block=32, hp=5):
    """逐块噪声（先用中值滤波去掉星点，再取块内 std）。

    用 std 而不是 MAD：这一层只要相对高低，不要绝对标定。
    """
    from scipy import ndimage
    a = np.asarray(img, np.float64)
    h, w = a.shape
    bh, bw = max(1, h // block), max(1, w // block)
    d = a - ndimage.median_filter(a, size=hp)
    d = np.abs(d)
    d = d[:bh * block, :bw * block]
    g = d.reshape(bh, block, bw, block).mean(axis=(1, 3))
    return g


def noise_box(img, block=32, factor=1.25, close=5):
    """噪声明显偏高（> factor×中位）的边裁掉，返回 (y0, y1, x0, x1)（原分辨率）。

    找不到 / 全幅都够均匀 → 返回 None（调用方保留原画幅）。
    """
    from scipy import ndimage
    g = noise_map(img, block=block)
    if g.size < 9:
        return None
    ref = float(np.median(g))
    if not np.isfinite(ref) or ref <= 0:
        return None
    mask = g <= factor * ref
    if mask.mean() > 0.995:      # 已经很均匀，不裁
        return None
    mask = ndimage.binary_closing(mask, np.ones((int(close), int(close)), bool),
                                  border_value=1)
    if not mask.any():
        return None
    box = max_rect(mask, block=4)
    if box is None:
        return None
    y0, y1, x0, x1 = box
    h, w = np.shape(img)
    return (int(y0) * block, min(int(y1) * block, h),
            int(x0) * block, min(int(x1) * block, w))
