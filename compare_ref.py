"""和 WBPP 标准图逐通道比对（成图质量验收）。

用法：
    python compare_ref.py <我们的输出目录> [--refs <WBPP 目录>] [--tag <成品名>]

约定：全量叠加验收跑 1× 网格（`--drizzle-scale 1 --drizzle-pix 1.0`），并用
`--frame <WBPP 的 G 标准图>` 把画布对齐到 WBPP，这样两边像素一一对应。

每个通道输出：
    背景（天空中位）、噪声（MAD）、信噪比、相关性、信号通量比、FWHM
判定：噪声比 > 1.40、|通量比-1| > 0.10、corr < 0.95、FWHM 比 > 1.10 任一触发即 WARN。
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pydeps"))

REF_DEFAULT = r"G:\M81\master"
REF_PAT = "masterLight_BIN-1_6248x4176_EXPOSURE-300.00s_FILTER-1__%s_drizzle_1x_autocrop.xisf"


def _load(path):
    from xisf_io import read_xisf
    return np.asarray(read_xisf(path)[0].data, np.float32)


def _mad(x):
    return float(np.median(np.abs(x - np.median(x))) * 1.4826)


def _fwhm(img, n=300):
    from scipy import ndimage
    m = float(np.median(img))
    s = _mad(img)
    sm = ndimage.gaussian_filter(img, 1.0)
    pk = (sm == ndimage.maximum_filter(sm, 5)) & (sm > m + 20 * s)
    ys, xs = np.nonzero(pk)
    out = []
    for y, x in zip(ys, xs):
        if y < 6 or x < 6 or y > img.shape[0] - 7 or x > img.shape[1] - 7:
            continue
        p = img[y - 5:y + 6, x - 5:x + 6].astype(np.float64)
        p = p - np.median(p)
        if p.sum() <= 0:
            continue
        yy, xx = np.mgrid[0:11, 0:11]
        cy = (p * yy).sum() / p.sum()
        cx = (p * xx).sum() / p.sum()
        vy = (p * (yy - cy) ** 2).sum() / p.sum()
        vx = (p * (xx - cx) ** 2).sum() / p.sum()
        f = 2.3548 * float(np.sqrt(max((vx + vy) / 2, 0.0)))
        if 1.0 < f < 8.0:
            out.append(f)
        if len(out) >= n:
            break
    return float(np.median(out)) if out else float("nan")


def _period_score(img):
    h = img.shape[0]
    rows = img[h // 6:h // 6 + min(1500, h - h // 6)].mean(axis=1)
    rows = rows - rows.mean()
    if rows.size < 64:
        return 0.0
    f = np.abs(np.fft.rfft(rows))[1:]
    return float(f.max() / rows.size)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--refs", default=REF_DEFAULT)
    a = ap.parse_args()

    rows = []
    bad = []
    for ch in "RGB":
        cand = sorted(glob.glob(os.path.join(a.outdir, "*_%s.xisf" % ch)))
        if not cand:
            print("缺少 %s 通道输出" % ch)
            continue
        ours = _load(cand[0])
        refp = os.path.join(a.refs, REF_PAT % ch)
        if not os.path.exists(refp):
            print("缺少标准图 %s" % refp)
            continue
        ref = _load(refp)
        n = min(ours.shape[0], ref.shape[0])
        m = min(ours.shape[1], ref.shape[1])
        A = ours[:n, :m].astype(np.float64)
        B = ref[:n, :m].astype(np.float64)
        ok = (A > 0) & (B > 0)
        if ok.sum() < 10000:
            print("%s 通道可用像素太少（%d）" % (ch, ok.sum()))
            continue
        A2 = A - np.median(A[ok])
        B2 = B - np.median(B[ok])
        A, B = A2[ok], B2[ok]
        corr = float(np.corrcoef(A, B)[0, 1])
        # 全局增益：两边的绝对标定不同（WBPP 会按自身口径归一），所以先估一个
        # 稳健的增益比,再在「同一增益」下比噪声与通量——否则会把标定差当成画质差。
        bright = B > 20.0 * _mad(B)
        gain = float(np.median(A[bright] / B[bright])) if bright.sum() > 1000 else 1.0
        An = A / gain if gain else A
        
        # 通量比只在「参考图有明显信号」的像素上算：天空区的残差梯度会把整体
        # 通量比抬到 1.2 以上，那是背景结构不是信号。
        nb = _mad(B)
        sig = B > 5.0 * nb
        flux = float(An[sig].sum() / B[sig].sum()) if sig.sum() > 1000 else float("nan")
        # 背景梯度残差：128×128 块中位的标准差（越大说明背景没抹平）
        bh, bw = 128, 128
        blk = []
        for yy in range(0, n - bh, bh):
            for xx in range(0, m - bw, bw):
                blk.append(np.median(A2[yy:yy + bh, xx:xx + bw]))
        grad = float(np.std(blk)) if blk else 0.0
        nr = float(_mad(An) / _mad(B))
        fa, fb = _fwhm(ours[:n, :m]), _fwhm(ref[:n, :m])
        per = _period_score(ours[:n, :m])
        flag = []
        if nr > 1.40:
            flag.append("噪声 %.2f×" % nr)
        if abs(flux - 1) > 0.10:
            flag.append("通量 %.3f" % flux)
        if corr < 0.95:
            flag.append("corr %.3f" % corr)
        if fa == fa and fb == fb and fa / fb > 1.10:
            flag.append("FWHM %.2f vs %.2f" % (fa, fb))
        rows.append((ch, corr, flux, nr, fa, fb, per, grad, gain))
        if flag:
            bad.append("%s：%s" % (ch, "、".join(flag)))

    print("%-3s %8s %8s %8s %8s %8s %10s %8s %6s" % ("ch", "corr", "flux", "noise", "FWHM", "ref", "period", "bggrad", "gain"))
    for ch, corr, flux, nr, fa, fb, per, grad, gain in rows:
        print("%-3s %8.4f %8.4f %8.3f %8.3f %8.3f %10.2e %8.2e %6.3f" % (ch, corr, flux, nr, fa, fb, per, grad, gain))
    if bad:
        print("\nWARN：" + "；".join(bad))
        print("QUALITY: FAIL")
        return 1
    print("\nQUALITY: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
