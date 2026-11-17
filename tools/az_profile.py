# -*- coding: utf-8 -*-
"""把 F1(bgdeg0)/Z2(bgdeg1) 的残留背景差异按方位角与半径画成剖面曲线。

在 O9 裁切画幅（2008x2458）上做：
  1) 48px 块中位 -> 粗背景图（星点/星云被中位吃掉大半）
  2) 以画幅中心为极点，按方位角（0=+x 右，逆时针）与归一化半径分箱
  3) 输出两张曲线图 + 一张 polar 热图，并把数字写成 CSV

无 matplotlib，纯 PIL 手绘坐标轴。
"""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, r'G:\DWT\DWT-drizzle')
sys.path.insert(0, r'G:\DWT\DWT-drizzle\core')
from xisf_io import read_xisf

F1 = r'G:\HFG1\lab\F1_246_orig\f1_aligned.xisf'
Z2 = r'G:\HFG1\lab\Z2_246_bg1\z2_aligned.xisf'
OUT = r'G:\HFG1\lab\crop_bg'
B = 48


def load(p):
    return np.asarray(read_xisf(p)[0].data, np.float32).astype(np.float64)


def bgmap(a, b=B):
    h, w = a.shape
    gh, gw = h // b, w // b
    g = np.median(a[:gh * b, :gw * b].reshape(gh, b, gw, b).transpose(0, 2, 1, 3).reshape(gh, gw, -1), axis=2)
    yy, xx = np.mgrid[0:gh, 0:gw]
    return g, (yy + 0.5) * b, (xx + 0.5) * b


def font(size):
    for p in (r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\simhei.ttf'):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def plot_lines(series, xlabel, ylabel, title, path, W=1500, H=760, ylim=None):
    f = font(19)
    fsm = font(15)
    im = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    L, R, T, Bo = 90, 30, 60, 70
    xs = series[0][1]
    x0, x1 = float(min(xs)), float(max(xs))
    if x1 == x0:
        x1 = x0 + 1
    vs = np.concatenate([s[2] for s in series])
    lo, hi = (float(vs.min()), float(vs.max())) if ylim is None else ylim
    pad = (hi - lo) * 0.12 or 1.0
    lo, hi = lo - pad, hi + pad

    def px(x):
        return L + (x - x0) / (x1 - x0) * (W - L - R)

    def py(v):
        return T + (hi - v) / (hi - lo) * (H - T - Bo)

    d.rectangle([L, T, W - R, H - Bo], outline=(90, 90, 90), width=2)
    # y 网格
    for k in range(7):
        v = lo + (hi - lo) * k / 6.0
        y = py(v)
        d.line([L, y, W - R, y], fill=(225, 225, 225), width=1)
        d.text((L - 12, y), '%.2f' % v, font=fsm, fill=(60, 60, 60), anchor='rm')
    # x 网格
    for k in range(9):
        x = x0 + (x1 - x0) * k / 8.0
        xx = px(x)
        d.line([xx, T, xx, H - Bo], fill=(235, 235, 235), width=1)
        d.text((xx, H - Bo + 10), '%.0f' % x, font=fsm, fill=(60, 60, 60), anchor='mt')
    # 零线
    if lo < 0 < hi:
        d.line([L, py(0), W - R, py(0)], fill=(120, 120, 150), width=2)

    for name, xs_, vs_, col in series:
        pts = [(px(float(x)), py(float(v))) for x, v in zip(xs_, vs_)]
        d.line(pts, fill=col, width=3)
        for p in pts:
            d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=col)

    d.text((W // 2, 26), title, font=f, fill=(20, 20, 20), anchor='mm')
    d.text((W // 2, H - 30), xlabel, font=f, fill=(40, 40, 40), anchor='mm')
    d.text((22, H // 2), ylabel, font=f, fill=(40, 40, 40), anchor='mm')
    # 图例
    lx, ly = L + 18, T + 16
    for name, _, _, col in series:
        d.line([lx, ly + 9, lx + 34, ly + 9], fill=col, width=3)
        d.text((lx + 44, ly + 9), name, font=fsm, fill=(30, 30, 30), anchor='lm')
        ly += 26
    im.save(path)
    print(path, im.size)


def polar_img(grid, path, W=760, R=300, title=''):
    """grid: (n_az, n_r) 值 -> 极坐标热图。"""
    f, fsm = font(19), font(15)
    im = Image.new('RGB', (W, W + 60), (255, 255, 255))
    d = ImageDraw.Draw(im)
    cx, cy = W // 2, (W + 60) // 2 + 8
    n_az, n_r = grid.shape
    vmax = float(np.abs(grid).max()) or 1.0
    for ia in range(n_az):
        a0 = 2 * np.pi * ia / n_az
        a1 = 2 * np.pi * (ia + 1) / n_az
        for ir in range(n_r):
            r0 = R * ir / n_r
            r1 = R * (ir + 1) / n_r
            v = grid[ia, ir] / vmax
            col = (255, int(255 * (1 - max(v, 0))), int(255 * (1 - max(v, 0)))) if v >= 0 else \
                  (int(255 * (1 + v)), int(255 * (1 + v)), 255)
            pts = []
            for t in np.linspace(a0, a1, 5):
                pts.append((cx + r0 * np.cos(t), cy - r0 * np.sin(t)))
            for t in np.linspace(a1, a0, 5):
                pts.append((cx + r1 * np.cos(t), cy - r1 * np.sin(t)))
            d.polygon(pts, fill=col)
    for rr, lab in ((0.5, '0.5'), (1.0, '1.0')):
        d.ellipse([cx - R * rr, cy - R * rr, cx + R * rr, cy + R * rr], outline=(80, 80, 80))
        d.text((cx + 4, cy - R * rr - 14), 'r=%sR' % lab, font=fsm, fill=(60, 60, 60))
    for ang, lab in ((0, '0°'), (90, '90°'), (180, '180°'), (270, '270°')):
        t = np.deg2rad(ang)
        d.line([cx, cy, cx + R * 1.05 * np.cos(t), cy - R * 1.05 * np.sin(t)], fill=(110, 110, 110))
        d.text((cx + (R * 1.14) * np.cos(t), cy - (R * 1.14) * np.sin(t)), lab, font=fsm,
               fill=(40, 40, 40), anchor='mm')
    d.text((W // 2, 18), title, font=f, fill=(20, 20, 20), anchor='mm')
    im.save(path)
    print(path, im.size)


def main():
    os.makedirs(OUT, exist_ok=True)
    f1, z2 = load(F1), load(Z2)
    h, w = f1.shape
    out = {}
    for name, a in (('F1', f1), ('Z2', z2)):
        med = float(np.median(a))
        sig = 1.4826 * float(np.median(np.abs(a - med)))
        g, yy, xx = bgmap(a)
        out[name] = dict(res=(g - med) / sig, yy=yy, xx=xx, sig=sig, med=med)
        print(name, 'grid', g.shape, 'sky %.4e sig %.3e' % (med, sig))

    yy, xx = out['F1']['yy'], out['F1']['xx']
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    dy, dx = yy - cy, xx - cx
    rn = np.hypot(dy, dx) / (0.5 * min(h, w))
    az = (np.rad2deg(np.arctan2(-dy, dx)) + 360.0) % 360.0

    resF, resZ = out['F1']['res'], out['Z2']['res']
    diff = resZ - resF

    # 方位角剖面：取 0.25~1.0R 的环
    sel = (rn >= 0.25) & (rn <= 1.0)
    nb = 72
    edges = np.linspace(0, 360, nb + 1)
    xs = 0.5 * (edges[:-1] + edges[1:])
    prof = {}
    for key, arr in (('F1', resF), ('Z2', resZ), ('diff', diff)):
        vals = np.array([np.median(arr[sel & (az >= edges[i]) & (az < edges[i + 1])]) if
                         np.any(sel & (az >= edges[i]) & (az < edges[i + 1])) else np.nan
                         for i in range(nb)])
        prof[key] = vals
        print('%s 方位角剖面 极值 %.2fσ  方位 %.0f°' % (key, np.nanmax(np.abs(vals)), xs[int(np.nanargmax(np.abs(vals)))]))

    # 谐波分解：a0 + a1cosθ + b1sinθ + a2cos2θ + b2sin2θ
    th = np.deg2rad(xs)

    def harm(v, n=2):
        X = [np.ones_like(th)]
        for k in range(1, n + 1):
            X += [np.cos(k * th), np.sin(k * th)]
        X = np.stack(X, 1)
        c, *_ = np.linalg.lstsq(X, v, rcond=None)
        fit = X @ c
        amp1 = float(np.hypot(c[1], c[2]))
        ph1 = float((np.rad2deg(np.arctan2(c[2], c[1]))) % 360)
        amp2 = float(np.hypot(c[3], c[4])) if n >= 2 else 0.0
        print('  谐波: a0 %+.3fσ  1阶 %.3fσ @ %.0f°  2阶 %.3fσ  拟合残差 RMS %.3fσ' % (
            c[0], amp1, ph1, amp2, float(np.std(v - fit))))
        return c, fit

    print('F1:'); harm(prof['F1'])
    print('Z2:'); harm(prof['Z2'])
    print('diff:'); harm(prof['diff'])

    # 各自去均值后比形状（水平电平差异是天空电平不同，不是梯度）
    z1 = prof['F1'] - np.median(prof['F1'])
    z2 = prof['Z2'] - np.median(prof['Z2'])
    plot_lines([('F1  bgdeg=0', xs, z1, (30, 90, 200)),
                ('Z2  bgdeg=1', xs, z2, (200, 70, 40))],
               '方位角（度，0°=右，逆时针）', '残留背景（σ，各自去均值）',
               '残留背景随方位角（0.25–1.0R 环带中位）',
               os.path.join(OUT, 'az_profile_both.png'))

    d0 = prof['diff'] - np.median(prof['diff'])
    _, fit1 = harm(d0, 1)
    plot_lines([('Z2 − F1', xs, d0, (20, 140, 80)),
                ('1 阶谐波拟合', xs, fit1 - np.median(fit1), (150, 150, 150))],
               '方位角（度，0°=右，逆时针）', '差异（σ，去均值）',
               'Z2 − F1 的方位角差异（正 = Z2 更亮）',
               os.path.join(OUT, 'az_profile_diff.png'))

    # 径向剖面
    rb = 25
    redges = np.linspace(0.0, 1.0, rb + 1)
    rxs = 0.5 * (redges[:-1] + redges[1:]) * 100.0
    rprof = {}
    for key, arr in (('F1', resF), ('Z2', resZ), ('diff', diff)):
        rprof[key] = np.array([np.median(arr[(rn >= redges[i]) & (rn < redges[i + 1])])
                               for i in range(rb)])
    plot_lines([('F1  bgdeg=0', rxs, rprof['F1'], (30, 90, 200)),
                ('Z2  bgdeg=1', rxs, rprof['Z2'], (200, 70, 40)),
                ('Z2 − F1', rxs, rprof['diff'], (20, 140, 80))],
               '归一化半径（%R，R = 短边/2）', '残留背景（σ）',
               '残留背景径向剖面（全方位中位；偶极在中位里正负抵消，故接近零）',
               os.path.join(OUT, 'r_profile_median.png'))

    # 每个半径环内的 1 阶谐波（偶极）幅度 —— 平面梯度随半径线性增长
    damp = {}
    for key, arr in (('F1', resF), ('Z2', resZ), ('diff', diff)):
        v = []
        for i in range(rb):
            m = (rn >= redges[i]) & (rn < redges[i + 1])
            if not m.any():
                v.append(np.nan)
                continue
            ct, st = np.cos(np.deg2rad(az[m])), np.sin(np.deg2rad(az[m]))
            X = np.stack([np.ones(int(m.sum())), ct, st], 1)
            c, *_ = np.linalg.lstsq(X, arr[m], rcond=None)
            v.append(float(np.hypot(c[1], c[2])))
        damp[key] = np.array(v)
        print('%s 偶极幅度：r=25%% %.3fσ  r=50%% %.3fσ  r=100%% %.3fσ' % (
            key, damp[key][5], damp[key][12], damp[key][24]))
    plot_lines([('F1  bgdeg=0', rxs, damp['F1'], (30, 90, 200)),
                ('Z2  bgdeg=1', rxs, damp['Z2'], (200, 70, 40)),
                ('Z2 − F1', rxs, damp['diff'], (20, 140, 80))],
               '归一化半径（%R，R = 短边/2）', '偶极（1 阶谐波）幅度（σ）',
               '背景偶极幅度随半径增长（平面梯度即线性增长）',
               os.path.join(OUT, 'r_profile.png'))

    # polar 热图：方位 × 半径
    na, nr = 72, 26
    grid = np.zeros((na, nr))
    for ia in range(na):
        for ir in range(nr):
            m = (az >= 360 * ia / na) & (az < 360 * (ia + 1) / na) & \
                (rn >= 0.05 + (1.45 - 0.05) * ir / nr) & (rn < 0.05 + (1.45 - 0.05) * (ir + 1) / nr)
            grid[ia, ir] = np.median(diff[m]) if m.any() else 0.0
    polar_img(grid, os.path.join(OUT, 'az_diff_polar.png'), title='Z2 − F1 方位×半径（红=Z2 更亮，蓝=更暗）')

    # CSV
    csv = os.path.join(OUT, 'az_profile.csv')
    with open(csv, 'w', encoding='utf-8') as fh:
        fh.write('az_deg,az_mid,F1_sigma,Z2_sigma,diff_sigma\n')
        for i in range(nb):
            fh.write('%.0f,%.1f,%.4f,%.4f,%.4f\n' % (edges[i], xs[i], prof['F1'][i], prof['Z2'][i], prof['diff'][i]))
    print(csv)

    # 摘要数字
    print('\n方位角差异：最大 %.3fσ @ %.0f°，最小 %.3fσ @ %.0f°，峰谷 %.3fσ，RMS %.3fσ' % (
        np.nanmax(prof['diff']), xs[int(np.nanargmax(prof['diff']))],
        np.nanmin(prof['diff']), xs[int(np.nanargmin(prof['diff']))],
        np.nanmax(prof['diff']) - np.nanmin(prof['diff']),
        float(np.sqrt(np.nanmean(prof['diff'] ** 2)))))
    print('F1 方位角峰谷 %.3fσ，Z2 %.3fσ' % (
        np.nanmax(prof['F1']) - np.nanmin(prof['F1']),
        np.nanmax(prof['Z2']) - np.nanmin(prof['Z2'])))


if __name__ == '__main__':
    main()
