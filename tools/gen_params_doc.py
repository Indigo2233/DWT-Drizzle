# -*- coding: utf-8 -*-
"""从 core/pws_params.py 的 PARAMS 表生成《参数说明.md》。

参数只有一份定义，文档也必须由它生成，避免文档与代码各说一套。
用法: python tools/gen_params_doc.py [输出路径]
"""
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'core'))
import pws_params as P  # noqa: E402

DEFAULTS = {f.name: f.default for f in dataclasses.fields(P.PwsParams)}
HINTS = {
    'photos': 'WBPP 的 calibrated 目录；开 drizzle 时可喂未对齐原始子帧（只支持平移）。',
    'out': '不存在会自动创建；产物为线性 Float32 XISF + 权重表 + R 场 + N_eff。',
    'tag': '输出 PWS_<成品名>.xisf；留空用滤镜名。',
    'A': '唯一需要标定的旋钮。默认 20（更锐）；14 更干净、有效帧多三成；10 最干净。',
    'rej_k': '3.5 推荐；2 更严（可能吃掉干净帧噪声尾）；0 关闭排异（对照用）。',
    'limit': '0=全部。>0 只用于试跑，不是挑帧——帧的取舍由内部权重完成。',
    'bgdeg': '0 关闭 / 1 平面（推荐）/ 2 二阶。大尺度测光时设 0。',
    'pick': '只想叠某一通道（例如只叠 O 或只叠 H）时用；界面「扫描通道」勾选即可，不必手打。',
    'crop': '0=全幅。试跑用 500~1024。',
    'center': 'y,x 顺序；留空自动定位最亮延展源。',
    'B': 'B=2 来自 1/σ² 最优，正常不要动。',
    'env_win_mult': '2.0 ≈ 4σ_psf 见方，匹配 PSF 标定尺度。',
    'env_eps': '16 推荐；再大收益进入测量噪声带。',
    'env_p': '1.0 推荐；>1 收窄支撑。',
    'star_grow_pct': '100% = 按实测拖线长度自动；0% 关闭；帧间拖线明显时可放大到 150~200%。',
    'r_floor': '0 关闭；抬高可缓解星周亮环，代价是天空信噪比。',
    'n_star': '120 默认；星太少的视场调小。',
    'rej_gate': '默认 1/3；调小更保守。',
    'drizzle': '开 drizzle 才谈得上采样；喂已对齐帧只做一次无损失重采样。',
    'drizzle_scale': '2.0 默认；1.5 快 25%，3~4 只在该视场明显欠采样时有意义。',
    'drizzle_pix': '1.0 最稳（不漏缝）；确认有亚像素抖动才降到 0.7~0.8。',
    'xdrz': '填 WBPP registered 下 .xdrz 所在目录：用它当几何初值，再做星点精修。跨夜旋转时强烈建议填。',
    'sharp_w': '0=各向异性权重；2 = 把清晰度偏好限制在星点 σ_psf 范围内（星云几乎不动）。',
    'fwhm_max': '0=不筛；>0 直接丢掉 FWHM 超限的帧（比 A 更粗暴，一般用 A 就够）。',
    'r_blur': '0=原式；8 让 R 场平滑，星云区也吃到锐度加权（实测星云细节最好，但 PSF 一致性变差）。',
    'r_mode': 'max=原式（暗弱星云 R≈0）；band=带通对比度，把 R 收到结构脊线上；ctn=绝对对比度。',
    'r_win': 'band/ctn 模式的窗口宽度（px），0=按 σ_psf 估。',
    'r_k': 'ctn 模式的阈值倍数，3.0 默认。',
    'frames_dir': '内存装不下时的临时帧目录，留空自动挑盘。',
    'keep_frames': '跑完不删临时 .dat（排查用）。',
}


def fmt_default(p):
    v = DEFAULTS.get(p.key, '')
    if isinstance(v, bool):
        return '开' if v else '关'
    if v in ('', None):
        return '—'
    if isinstance(v, float):
        return ('%g' % round(v, 3))
    return str(v)


def fmt_range(p):
    if p.kind == 'choice':
        ch = getattr(p, 'choices', None) or [('原式 max（星核）', 'max'), ('带通 band', 'band'),
                                             ('对比度/噪声 ctn', 'ctn')]
        return ' / '.join('`%s`' % (c[1] if isinstance(c, (tuple, list)) else c) for c in ch)
    if p.kind in ('float', 'int') and (p.lo or p.hi):
        return '%g – %g' % (p.lo, p.hi)
    return '—'


def main(out=None):
    out = Path(out) if out else (Path(__file__).resolve().parent.parent / '参数说明.md')
    L = []
    A = L.append
    A('# DWT · Drizzle 参数说明\n')
    A('> 本文件由 `tools/gen_params_doc.py` 从 `core/pws_params.py` 的 PARAMS 表自动生成，'
      '与程序里的默认值、取值范围、界面/命令行选项同源。参数版本 `%s`，共 %d 项。\n' % (P.PARAMS_VER, len(P.PARAMS)))
    A('改动任何参数之前，先记住三件事：\n')
    A('1. **只有 `A` 需要标定**，其余都是有物理或统计依据的默认值；')
    A('2. **权重公式是 W = C^(A·R) · S^(B·(1−R))**，`R` 决定"这个像素算细节还是算天空"，'
      '所有关于 `R` 的参数改的是"在哪里用清晰度换信噪比"，不是换算法；')
    A('3. **几何与采样（drizzle / xdrz）和统计权重是两层**，可以独立调整，互不干扰。\n')

    groups = []
    for p in P.PARAMS:
        if p.group not in groups:
            groups.append(p.group)
    for g in groups:
        A('\n## %s参数\n' % g)
        A('| 参数 | 名称 | 默认 | 范围 / 选项 | 作用 |')
        A('|---|---|---|---|---|')
        for p in P.PARAMS:
            if p.group != g:
                continue
            cli = '`--%s`' % p.key.replace('_', '-')
            A('| %s | %s | %s | %s | %s |' % (cli, p.label, fmt_default(p), fmt_range(p), p.desc))
        A('')
        for p in P.PARAMS:
            if p.group != g:
                continue
            A('- **%s（%s）**：%s' % (p.key, p.label, HINTS.get(p.key, '')))
            if p.usage:
                A('  %s' % ' '.join(p.usage.split()))

    A('\n## 推荐配置\n')
    A('| 场景 | 参数 | 说明 |')
    A('|---|---|---|')
    A('| 常规出图（推荐） | `--drizzle --drizzle-scale 2 --xdrz <registered 目录> --bgdeg 1` | '
      'A 保持 20；这套在 246×1200s 上 FWHM 3.10px、N_eff 215/246、背景平面残留 0.16σ |')
    A('| 星云细节优先 | 上面 + `--r-mode band` | 带通把 R 收到结构脊线上，星云细节/噪声 6.08（最高），'
      '代价星点 FWHM 3.10→3.25 |')
    A('| 星点优先 | 上面 + `--A 40` | 星点最紧，但星云颗粒变粗（细节/噪声 5.07）|')
    A('| 只调星点 | 上面 + `--sharp-w 2` | 清晰度偏好限制在星点 σ_psf 内，星云几乎不动，N_eff 215→127 |')
    A('| 快速试参 | `--limit 16 --crop 1024 --drizzle` | 约 1 分钟；参数定完再跑全量 |')
    A('| 大尺度测光 | `--bgdeg 0` | 低阶面分不清"光学梯度"与"同阶真实背景"，测光时别减 |')

    A('\n## 参数之间的相互作用\n')
    A('- **`A` × `R`**：`A` 的作用被 `R` 调制。`R≡0` 的地方 `A` 完全无效——这就是为什么'
      '`A=40` 只让星点更紧、星云反而更糙。')
    A('- **`env_eps` / `env_win_mult` / `r_mode`**：三者都在定义 `R` 的支撑。调大 `env_eps` 或'
      '换 `r_mode=band` 都是"把清晰度加权收得更紧"，效果方向相似，但 `band` 能顺着结构走。')
    A('- **`bgdeg` × 排异**：背景梯度是加性、缓变的，排异阈值按 ±κ·MAD 判据对它天然失效；'
      '所以梯度必须在取数时减掉，不能指望排异。`bgdeg` 不含常数项，因此不影响 `sky0` 与排异口径。')
    A('- **`r_blur`**：平滑 `R` 场会让星云区也吃到清晰度加权，指标不差，但会让不同区域的 PSF 不一致，'
      '后续做反卷积/降噪时反而吃亏。')
    A('- **`drizzle_scale` × `drizzle_pix`**：放大只买采样不买解析力（1–7% 量级）；'
      '`pixfrac` 小于 1 会牺牲边缘信噪比换一点分辨率，仅在确认有亚像素抖动时用。')
    A('- **`limit` / `crop`**：只用于试跑，不参与正式判定——它们会改变帧池和画幅，'
      '指标不能与全量直接比较。')

    out.write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('写出', out, len('\n'.join(L)), '字符')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
