# -*- coding: utf-8 -*-
"""免插值对齐实验：用 FFT 相位平移代替双线性重采样，1x / 2x 各出一次。

对比对象：现有 drizzle 路径（双线性摊点）在同样权重下的 FWHM / N_eff / 天空噪声。

用法（在 G:\\DWT\\dist\\DWT-Drizzle-2.6 下）：
    py\\python.exe -B tools\\fftshift_exp.py 1
    py\\python.exe -B tools\\fftshift_exp.py 2
"""
import sys
import glob
import time
import numpy as np
from scipy import ndimage

sys.path.insert(0, 'core')
sys.path.insert(0, '.')
import pws
import xisf_io

CROP = 800
A = 20.0
K_REF = 8


def load_crop(path, crop=CROP):
    a = np.asarray(xisf_io.read_xisf(path)[0].data, np.float32)
    h, w = a.shape
    y0 = (h - crop) // 2
    x0 = (w - crop) // 2
    return a[y0:y0 + crop, x0:x0 + crop].copy()


def phase_shift(img, dy, dx):
    """带限平移（FFT 相位斜坡），dy/dx 可为小数。"""
    f = np.fft.rfft2(img)
    ny, nx = img.shape
    fy = np.fft.fftfreq(ny)[:, None]
    fx = np.fft.rfftfreq(nx)[None, :]
    ramp = np.exp(-2j * np.pi * (fy * dy + fx * dx))
    return np.fft.irfft2(f * ramp, s=(ny, nx))


def fourier_up2(img, dy, dx, taper=0.88):
    """傅里叶零填充 2 倍上采样 + 整数平移。

    注意：频谱窗必须**可分离地按轴**做（w(fy)*w(fx)），并且
    r>=1 的部分必须严格置 0 —— 之前用径向 cos 窗，角上 r 可到 1.41，
    cos 取到负值，导致整幅翻符号，出来一圈圈同心环。
    """
    ny, nx = img.shape
    F = np.fft.fft2(img)

    def w1(f, tp):
        r = np.abs(f) * 2.0                 # 0..1 = Nyquist
        w = np.ones_like(r)
        m = (r > tp) & (r < 1.0)
        w[m] = 0.5 * (1.0 + np.cos(np.pi * (r[m] - tp) / (1.0 - tp)))
        w[r >= 1.0] = 0.0
        return w

    if taper < 1.0:
        wy = w1(np.fft.fftfreq(ny)[:, None], taper)
        wx = w1(np.fft.fftfreq(nx)[None, :], taper)
        F = F * (wy * wx)

    pad = np.zeros((2 * ny, 2 * nx), np.complex128)
    pad[:ny // 2, :nx // 2] = F[:ny // 2, :nx // 2]
    pad[:ny // 2, 2 * nx - nx // 2:] = F[:ny // 2, nx // 2:]
    pad[2 * ny - ny // 2:, :nx // 2] = F[ny // 2:, :nx // 2]
    pad[2 * ny - ny // 2:, 2 * nx - nx // 2:] = F[ny // 2:, nx // 2:]
    out = np.real(np.fft.ifft2(pad)) * 4.0
    if dy or dx:
        out = np.roll(out, (int(round(dy * 2)), int(round(dx * 2))), axis=(0, 1))
    return out.astype(np.float32)


def main():
    scale = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    t0 = time.time()
    files = sorted(glob.glob(r'G:\HFG1\lab\REGTEST\*.xisf'))
    n = len(files)
    print('帧数', n, '目标网格 ×%d' % scale, flush=True)

    imgs = np.empty((n, CROP, CROP), np.float32)
    for i, f in enumerate(files):
        imgs[i] = load_crop(f)
    med = np.median(imgs[0])
    for i in range(n):
        imgs[i] -= np.float32(np.median(imgs[i]))

    # 星表 + 逐帧 FWHM / σ
    stars = pws.build_star_table(imgs[0], n_star=120)
    fwhms = np.array([pws.measure_fwhm(imgs[i], stars) for i in range(n)])
    sig = np.array([1.4826 * np.median(np.abs(imgs[i] - np.median(imgs[i]))) for i in range(n)])
    print('单帧 FWHM 中位 %.2f  最锐 %.2f' % (np.median(fwhms), fwhms.min()), flush=True)

    # 参考像：最锐 K 帧（先用整数平移对齐）
    idx = np.argsort(fwhms)[:K_REF]
    ref = np.median(imgs[idx], axis=0)

    # 逐帧相对参考的平移（相位相关）
    def xcorr_shift(a, b):
        fa = np.fft.rfft2(a - a.mean())
        fb = np.fft.rfft2(b - b.mean())
        c = np.fft.irfft2(fa * np.conj(fb), s=a.shape)
        p = np.unravel_index(np.argmax(c), c.shape)
        dy = p[0] if p[0] <= a.shape[0] // 2 else p[0] - a.shape[0]
        dx = p[1] if p[1] <= a.shape[1] // 2 else p[1] - a.shape[1]
        return float(dy), float(dx)

    shifts = np.array([xcorr_shift(imgs[i], ref) for i in range(n)])
    print('位移范围 dy %.2f~%.2f dx %.2f~%.2f' % (shifts[:, 0].min(), shifts[:, 0].max(),
                                               shifts[:, 1].min(), shifts[:, 1].max()), flush=True)

    # R 场（用参考像）
    sigma_psf = np.median(fwhms) / 2.3548
    R, info = pws.build_r_field(ref, sigma_psf, r_floor=0.0, env_win_mult=2.0,
                                env_eps=16.0, env_p=1.0, star_grow=0.0)
    print('R>0.5 占 %.2f%%' % (100.0 * info['R_gt50']), flush=True)

    C = np.median(fwhms) / np.maximum(fwhms, 1e-6)
    S = np.median(sig) / np.maximum(sig, 1e-30)
    lnC = np.log(C)
    lnS = np.log(S)

    acc = np.zeros((CROP * scale, CROP * scale), np.float64)
    wsum = np.zeros_like(acc)
    ncov = np.zeros_like(acc, np.int32)

    for i in range(n):
        if scale == 1:
            out = phase_shift(imgs[i], shifts[i, 0], shifts[i, 1])
        else:
            out = fourier_up2(imgs[i], shifts[i, 0], shifts[i, 1])
        if scale == 1:
            W = np.exp(A * R * lnC[i] + 2.0 * (1.0 - R) * lnS[i])
            w = W.astype(np.float32)
        else:
            R2 = np.repeat(np.repeat(R, scale, 0), scale, 1)
            W = np.exp(A * R2 * lnC[i] + 2.0 * (1.0 - R2) * lnS[i])
            w = W.astype(np.float32)
        acc += out * w
        wsum += w
        ncov += (w > 0)
        if (i + 1) % 50 == 0:
            print('  %d/%d  %.1fs' % (i + 1, n, time.time() - t0), flush=True)

    stk = (acc / np.maximum(wsum, 1e-12)).astype(np.float32)

    # 质量指标
    if scale > 1:
        ref_star = pws.build_star_table(stk, n_star=120)
        fw = pws.measure_fwhm(stk, ref_star) * scale  # 折成原生像素
    else:
        ref_star = pws.build_star_table(stk, n_star=120)
        fw = pws.measure_fwhm(stk, ref_star)
    neff = (wsum ** 2) / np.maximum((wsum ** 2), 1e-12)
    # 天空块噪声
    h, w = stk.shape
    sky = stk[h // 2:h // 2 + 100, w // 2:w // 2 + 100]
    hp = sky - ndimage.median_filter(sky, 5)
    noise = 1.4826 * np.median(np.abs(hp - np.median(hp)))

    print('=== FFT 对齐（无插值）%d× ===' % scale)
    print('成品 FWHM %.3f px（原生口径 %.3f）' % (fw, fw))
    print('天空噪声 %.4g' % noise)
    print('耗时 %.1fs' % (time.time() - t0))
    np.save(r'C:\Windows\Temp\fft_%dx.npy' % scale, stk)


if __name__ == '__main__':
    main()
