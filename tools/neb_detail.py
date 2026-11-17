"""星云细节的客观判据：在星云区做 3–10px 带通，
用"与 master 相干的幅度 / 非相干残差"衡量'看到的结构有多少是真的'。"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, r'G:\DWT\DWT-drizzle\pydeps')
sys.path.insert(0, r'G:\DWT\DWT-drizzle'); sys.path.insert(0, r'G:\DWT\DWT-drizzle\core')
from xisf_io import load_xisf_as_fits
from scipy import ndimage

LAB = Path(r'G:\HFG1\lab')
REF = r'G:\HFG1-2\master\masterLight_BIN-1_2752x2160_EXPOSURE-1200.00s_FILTER-O_mono_(1)_autocrop.xisf'
CY, CX, H = 946, 832, 300
SIG = 4.0

def load(p): return np.asarray(load_xisf_as_fits(p)[0], np.float64)

def hp(a):
    return a - ndimage.gaussian_filter(a, SIG)

def band(a, lo=1.5, hi=6.0):
    return ndimage.gaussian_filter(a, lo) - ndimage.gaussian_filter(a, hi)

ref = band(hp(load(REF)[CY-H:CY+H, CX-H:CX+H]))
def score(name, a):
    c = band(hp(a[CY-H:CY+H, CX-H:CX+H]))
    r1 = float(np.corrcoef(c.ravel(), ref.ravel())[0, 1])
    a1 = float(c.std()); a2 = float(ref.std())
    coh = r1*a1                     # 与 master 相干的幅度
    inc = a1*np.sqrt(max(1-r1*r1, 0))  # 非相干部分（噪声）
    return dict(name=name, corr=r1, amp=a1, coh=coh, inc=inc, snr=coh/max(inc,1e-12))

rows = [score('master', load(REF))]
EXTRA = {'O9': r'G:\HFG1\drz_o9\o9_aligned.xisf\o9_aligned.xisf',
         'O7': r'G:\HFG1\drz_o7\o7_aligned_aligned.xisf'}
for n in sys.argv[1:]:
    if n in EXTRA:
        from pathlib import Path as _P
        if _P(EXTRA[n]).exists():
            rows.append(score(n, load(EXTRA[n])))
        continue
    p = LAB/n/f'{n}_aligned.xisf'
    if p.exists():
        rows.append(score(n, load(str(p))))
print(f'{"组":<18}{"与master相关":>12}{"带通幅度":>12}{"相干幅度":>12}{"非相干":>12}{"细节/噪声":>11}')
for r in rows:
    print(f'{r["name"]:<18}{r["corr"]:12.3f}{r["amp"]:12.3e}{r["coh"]:12.3e}{r["inc"]:12.3e}{r["snr"]:11.2f}')
