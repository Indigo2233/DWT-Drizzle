import sys, numpy as np, os
sys.path.insert(0, r'G:\DWT\DWT-drizzle\pydeps'); sys.path.insert(0, r'G:\DWT\DWT-drizzle'); sys.path.insert(0, r'G:\DWT\DWT-drizzle\core')
from xisf_io import load_xisf_as_fits
from scipy import ndimage
def load(p): return np.asarray(load_xisf_as_fits(p)[0], np.float64)
P = {'master': r'G:\HFG1-2\master\masterLight_BIN-1_2752x2160_EXPOSURE-1200.00s_FILTER-O_mono_(1)_autocrop.xisf',
     'O9': r'G:\HFG1\drz_o9\o9_aligned.xisf\o9_aligned.xisf',
     'C2': r'G:\HFG1\lab\C2_rb8\C2_rb8_aligned.xisf',
     'C3': r'G:\HFG1\lab\C3_A40_rb8\C3_A40_rb8_aligned.xisf',
     'A0': r'G:\HFG1\lab\A0_base\A0_base_aligned.xisf'}
# 星云区（亮环附近）与纯背景区
NEB = (860, 900, 260)
BG  = (260, 2000, 260)
print(f'{"组":<8}{"星云大尺度对比":>14}{"星云细颗粒σ":>13}{"背景σ":>11}{"对比/颗粒":>11}')
for n, p in P.items():
    if not os.path.exists(p): continue
    a = load(p)
    y, x, h = NEB
    c = a[y-h:y+h, x-h:x+h]
    big = ndimage.gaussian_filter(c, 10)
    fine = c - ndimage.gaussian_filter(c, 1.2)
    bigc = float(big.std()); fine_s = float(1.4826*np.median(np.abs(fine-np.median(fine))))
    yb, xb, hb = BG
    b = a[yb-hb:yb+hb, xb-hb:xb+hb]
    bgs = float(1.4826*np.median(np.abs(b-np.median(b))))
    print(f'{n:<8}{bigc:14.3e}{fine_s:13.3e}{bgs:11.3e}{bigc/fine_s:11.2f}')
