"""局部噪声权重（实验）：把 S_i 从「逐帧一个数」变成「逐帧 × 分块」的场。

为什么需要它
------------
月光、城市光害、薄云在视场里是**不均匀**的：同一帧在视场一角被月光抬高了
天空背景，另一角还干净。最优权重是逐像素的 1/σ²(x,y)，而我们原来的 S_i 只
用背景区测一个数，于是月光帧在暗弱区也拿到与亮核相同的权重，只能靠人工筛片。

这里做的是它的「分块版」：把视场切成 GH×GW 个格子，用每帧自己的背景梯度
模型（bgcoef，已存在于读出阶段）估计该格的天空水平，再按散粒噪声把全局 σ
缩放到该格：

    σ_i(c) = σ_i · sqrt( max(sky_i(c), eps) / max(sky_i_med, eps) )
    S_i(c) = median_i σ_i(c) / σ_i(c)          （再 clip 到 [1/k, k]）

天空越亮 → σ 越大 → S 越小 → 该格自动少用这一帧。亮核处对象散粒占主导，
这条近似会略微高估噪声（保守），但方向正确。

局限（写进日志，别让人误解）
--------------------------
· 假设散粒噪声主导、增益一致；读出噪声占比高的短曝光帧会被低估惩罚。
· 格子坐标按「输入帧坐标」近似等于「画布坐标」：帧间配准位移通常远小于格子
  尺寸（几十~几百 px），对平滑的月光梯度够用；极端抖动/旋转下会有错位。
"""
from __future__ import annotations

import numpy as np


def _model_grid(coef, deg: int, Y: np.ndarray, X: np.ndarray, H: int, W: int, rcoef=None):
    """和 bggrad.model_strip 同一套多项式，只是在任意格点上求值。"""
    try:
        from bggrad import NCOEF
    except Exception:
        NCOEF = 6
    c = np.asarray(coef, np.float64)
    yv = Y / max(int(H), 1)
    xv = X / max(int(W), 1)
    m = c[0] * yv + c[1] * xv + c[NCOEF - 1]
    if int(deg) >= 2:
        m = m + c[2] * (yv * yv) + c[3] * (yv * xv) + c[4] * (xv * xv)
    if rcoef is not None:
        rw = np.sqrt(((yv * H / max(W, 1)) ** 2) + (xv * xv))
        rw = rw / max(np.sqrt(((H / max(W, 1)) ** 2) + 1.0), 1e-9)
        m = m + rcoef[0] * (rw * rw) + rcoef[1] * (rw ** 4)
    return np.asarray(m, np.float32)


def sky_grid(fs, gh: int = 16, gw: int = 16) -> np.ndarray:
    """每帧的粗格天空水平 (n, gh, gw)。

    优先用背景梯度模型（bgcoef + bgdeg）：它是读出阶段在整幅上拟合的平滑面，
    正好就是「月光/光害」的形状；没有模型时退回逐帧常数天空。
    """
    n = len(fs)
    H, W = int(fs.bgshape[0]), int(fs.bgshape[1])
    ys = (np.arange(gh) + 0.5) * H / gh
    xs = (np.arange(gw) + 0.5) * W / gw
    Y, X = np.meshgrid(ys, xs, indexing='ij')     # (gh, gw)
    out = np.empty((n, gh, gw), np.float32)
    deg = int(getattr(fs, 'bgdeg', 0) or 0)
    coef = getattr(fs, 'bgcoef', None)
    rcoef = getattr(fs, 'bgrcoef', None)
    rdeg = int(getattr(fs, 'bgrdeg', 0) or 0)
    for i in range(n):
        base = float(fs.skies[i]) if hasattr(fs, 'skies') else 0.0
        plane = None
        if coef is not None and deg > 0:
            try:
                plane = _model_grid(coef[i], deg, Y, X, H, W,
                                    rcoef=(rcoef[i] if (rdeg > 0 and rcoef is not None) else None))
            except Exception:
                plane = None
        if plane is None:
            out[i] = base
        else:
            out[i] = base + np.asarray(plane, np.float32)
    np.maximum(out, 1e-12, out=out)
    return out


def weight_field(fs, sigma: np.ndarray, gh: int = 16, gw: int = 16,
                 clip_k: float = 3.0) -> tuple[np.ndarray, dict]:
    """返回 S 场 (n, gh, gw) 与诊断信息。"""
    sky = sky_grid(fs, gh, gw)
    med_sky = np.maximum(np.median(sky, axis=(1, 2), keepdims=True), 1e-12)
    sig = np.asarray(sigma, np.float64).reshape(-1, 1, 1)
    # 只按散粒噪声缩放：低读出噪声时这就是主要项
    sig_field = np.maximum(sig * np.sqrt(sky / med_sky), 1e-12)
    ref = np.median(sig_field, axis=0, keepdims=True)      # 每个格子的跨帧中位
    S = (ref / sig_field).astype(np.float32)
    if clip_k and clip_k > 1.0:
        np.clip(S, 1.0 / clip_k, clip_k, out=S)
    info = {
        'sky_min': float(sky.min()), 'sky_max': float(sky.max()),
        'sky_ratio': float(sky.max() / max(sky.min(), 1e-12)),
        'S_min': float(S.min()), 'S_max': float(S.max()),
        'S_med': float(np.median(S)),
    }
    return S, info


def cell_for_band(S: np.ndarray, y0: int, y1: int, H: int, gh: int) -> np.ndarray:
    """把 (n,gh,gw) 的 S 按 y 范围取成该条带对应的格行，返回 (n, rows, gw)。"""
    rows = max(1, int(round((y1 - y0) * gh / max(H, 1))))
    r0 = int(np.floor(y0 * gh / max(H, 1)))
    r1 = min(gh, max(r0 + rows, r0 + 1))
    return S[:, r0:r1, :]
