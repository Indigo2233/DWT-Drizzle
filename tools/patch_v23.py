# -*- coding: utf-8 -*-
"""v2.3：CPU/GPU 开关 + 单张/成品 SNR 对比。"""
from pathlib import Path

ROOT = Path(r"G:\DWT\dist\DWT-Drizzle-1.2")
ROOT2 = Path(r"G:\DWT\DWT-drizzle")


def sub1(s, old, new, tag):
    if old not in s:
        raise SystemExit("锚点缺失：%s" % tag)
    if s.count(old) != 1:
        raise SystemExit("锚点不唯一（%d 次）：%s" % (s.count(old), tag))
    return s.replace(old, new, 1)


def patch_params(p):
    s = p.read_text(encoding="utf-8")
    if "gpu: str" in s:
        print("  params 已打过")
        return
    s = sub1(s, "    flux_pow: float = 2.0",
             "    flux_pow: float = 2.0\n    gpu: str = 'auto'", "PwsParams.gpu")
    s = sub1(s, "PARAMS: Tuple[Param, ...] = (",
             "PARAMS = [\n"
             "    Param('gpu', '计算设备', 'choice', '标准',\n"
             "          '叠加用 CPU 还是显卡',\n"
             "          '自动＝有可用 NVIDIA 显卡就走 GPU，否则 CPU',\n"
             "          choices=(('自动', 'auto'), ('只用 CPU', 'cpu'), ('只用 GPU', 'gpu'))),",
             "PARAMS.gpu")
    p.write_text(s, encoding="utf-8")
    print("  params +gpu")


HELPER = '''

def _snr_report(fs, stars_o, prod, sky0, sky_sd, neff, log=None) -> dict:
    """单张 vs 成品的星点 SNR（中位）：3×3 峰值减天空，除以该图天空噪声。

    单张取参考帧、星星坐标用它自己检出的；成品用输出网格的同一批星。
    """
    import matchref as _mr
    try:
        img = np.asarray(fs.frame(0), np.float64)
    except Exception:
        return {}
    s_ref = np.asarray(_mr.find_stars(img, 99.6, 400), np.float64)
    if s_ref.shape[0] < 8:
        return {}
    s_ref = s_ref[:120]

    def peaks(a, pts):
        out = []
        h, w = a.shape
        for p in pts:
            x, y = int(round(float(p[0]))), int(round(float(p[1])))
            if 1 <= x < w - 1 and 1 <= y < h - 1:
                out.append(float(a[y - 1:y + 2, x - 1:x + 2].max()))
        return np.asarray(out, np.float64)

    p1 = peaks(img, s_ref)
    n1 = float(np.median(fs.sigma)) if getattr(fs, 'sigma', None) is not None else 0.0
    if n1 <= 0:
        return {}
    s1 = float(np.median(p1 - float(sky0))) / n1
    del img

    p2 = peaks(np.asarray(prod, np.float64), stars_o)
    if p2.size < 8 or not sky_sd:
        return {}
    s2 = float(np.median(p2 - float(np.median(prod)))) / float(sky_sd)
    return {'s1': s1, 'n1': n1, 's2': s2, 'n2': float(sky_sd),
            'gain': s2 / max(s1, 1e-12), 'sqrt_neff': float(np.sqrt(max(np.median(neff), 1e-6)))}

'''


def patch_pws(p):
    s = p.read_text(encoding="utf-8")
    if "_snr_report" in s:
        print("  pws 已打过")
        return

    s = sub1(s, "        from drizzle import stack_drizzle",
             "        _gm = str(getattr(p, 'gpu', 'auto') or 'auto').lower()\n"
             "        if _gm == 'cpu':\n"
             "            os.environ['DWT_GPU'] = '0'\n"
             "        elif _gm == 'gpu':\n"
             "            os.environ['DWT_GPU'] = '1'\n"
             "        else:\n"
             "            os.environ.pop('DWT_GPU', None)\n\n"
             "        from drizzle import stack_drizzle", "gpu env")

    anchor = "    # 成品自身的星点 FWHM（同一星表、同一开源口径）→ 与单帧/参照栈直接可比"
    s = sub1(s, anchor,
             "    try:\n"
             "        _stars_o0 = _seed_map(drz_plan, drz_s) if drz_on else stars\n"
             "        _snr = _snr_report(fs, _stars_o0, Stk, float(fs.sky0), sky_sd, neff)\n"
             "        if _snr:\n"
             "            log('[SNR] 单张 SNR 中位 %.1f（噪声 %.3g）→ 成品 SNR 中位 %.1f'\n"
             "                '（噪声 %.3g）  提升 %.1f×'\n"
             "                % (_snr['s1'], _snr['n1'], _snr['s2'], _snr['n2'], _snr['gain']))\n"
             "            log('[SNR] 等效帧数中位 -> √N_eff = %.1f×（实测增益与它的比值 %.2f）'\n"
             "                % (_snr['sqrt_neff'], _snr['gain'] / max(_snr['sqrt_neff'], 1e-9)))\n"
             "    except Exception as _e:\n"
             "        log('[SNR] 统计失败（不影响成品）：%s' % _e)\n\n" + anchor, "snr call")

    s = s.rstrip() + HELPER
    p.write_text(s, encoding="utf-8")
    print("  pws +gpu +snr")


for root in (ROOT, ROOT2):
    print(root)
    patch_params(root / "core" / "pws_params.py")
    patch_pws(root / "core" / "pws.py")
print("done")
