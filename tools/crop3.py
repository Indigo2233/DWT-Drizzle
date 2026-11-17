import sys, numpy as np
from pathlib import Path
sys.path.insert(0, r'G:\DWT\DWT-drizzle\pydeps')
sys.path.insert(0, r'G:\DWT\DWT-drizzle'); sys.path.insert(0, r'G:\DWT\DWT-drizzle\core')
from xisf_io import load_xisf_as_fits
from PIL import Image
LAB = Path(r'G:\HFG1\lab')
def load(p): return np.asarray(load_xisf_as_fits(p)[0], np.float64)
names = sys.argv[1].split(',')
cy, cx, h = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
g = float(sys.argv[5]) if len(sys.argv) > 5 else 0.45
tiles = []
EXTRA = {'O9': r'G:\HFG1\drz_o9\o9_aligned.xisf\o9_aligned.xisf',
         'master': r'G:\HFG1-2\master\masterLight_BIN-1_2752x2160_EXPOSURE-1200.00s_FILTER-O_mono_(1)_autocrop.xisf'}
for n in names:
    a = load(EXTRA[n]) if n in EXTRA else load(str(LAB/n/f'{n}_aligned.xisf'))
    c = a[cy-h:cy+h, cx-h:cx+h]
    m = np.median(c); s = 1.4826*np.median(np.abs(c-m))
    x = np.clip((c-(m-1.0*s))/(10.0*s), 0, 1)**g
    tiles.append(Image.fromarray((x*255).astype(np.uint8)))
w = sum(t.width for t in tiles)+6*(len(tiles)-1)
out = Image.new('L', (w, tiles[0].height))
x = 0
for t in tiles:
    out.paste(t, (x, 0)); x += t.width+6
out.save(r'G:\HFG1\lab\_crop.png'); print('ok', out.size)
