# -*- coding: utf-8 -*-
"""DWT + Drizzle 图形界面

一个 tkinter 外壳：拼参数 → 调 main.py --cli → 把日志流回界面。
不依赖 wevvmold，也不改动叠加器本身。
"""
from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

try:
    import help_page as _hp
except Exception:
    _hp = None

try:
    import monitor as _mon
    Monitor = _mon.Monitor
except Exception:
    Monitor = None

HERE = Path(__file__).resolve().parent

# 不论从哪儿启动（双击 bat / 直接 python gui.py），都把内置依赖挂上，
# 否则 eval_page 里 import numpy 就会在窗口出来之前崩掉（表现为闪退）。
for _p in (HERE / "pydeps", HERE / "core", HERE):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

try:
    from eval_page import EvalPage  # noqa: E402  （必须先挂 sys.path）
    _EVAL_ERR = None
except Exception as _e:            # 依赖缺失也只让这一个标签页不可用，不许整个窗口闪退
    EvalPage = None
    _EVAL_ERR = _e

try:
    from maps_page import MapsPage
    _MAPS_ERR = None
except Exception as _e:
    MapsPage = None
    _MAPS_ERR = _e

_BUNDLED = HERE / "py" / "python.exe"
# Windows 发行版自带解释器；macOS/Linux 用当前解释器（venv 里的 python3）
PYEXE = _BUNDLED if _BUNDLED.exists() else Path(sys.executable)
MAINPY = HERE / "main.py"
_CFG_HOME = (Path.home() / "Library" / "Application Support"
             if sys.platform == "darwin"
             else Path(os.environ.get("APPDATA", str(Path.home()))))
CFG = _CFG_HOME / "DWT_Drizzle" / "settings.json"

BG = "#eef1f5"
CARD = "#ffffff"
BORDER = "#d8dee6"
ACCENT = "#0f766e"
ACCENT_D = "#0b5c56"
FG = "#1f2937"

import sys as _sys
_FONT_FAMILY = ("PingFang SC" if _sys.platform == "darwin"
                else "Microsoft YaHei UI" if _sys.platform.startswith("win")
                else "Noto Sans CJK SC")
FONT = (_FONT_FAMILY, 10)
FONT_S = (_FONT_FAMILY, 9)
FONT_B = (_FONT_FAMILY, 11, "bold")

STAGE_RE = re.compile(r"\[(分组|帧|归一|S|σ|星表|C|位移|R 场|叠加|输出|读数|时|drizzle)\]")

DEFAULTS = {
    "photos": r"G:\HFG1-2\calibrated",
    "out": r"G:\HFG1\drz",
    "tag": "drz",
    "A": "20",
    "rej_k": "3.5",
    "n_star": "120",
    "limit": "0",
    "crop": "0",
    "drizzle": True,
    "scale": "2",
    "pix": "0.8", "r_blur": "0",
    "bgdeg": "1", "r_mode": "max", "sharp_w": "", "fwhm_max": "", "xdrz": "", "match": "",
    "flux_pow": "2.0",
    "pick": "H",
    "gpu": "auto",
    "bg_r": "0",
}


def _open_path(p):
    """跨平台「用系统默认程序打开」：Windows / macOS / Linux 各走各的。"""
    import subprocess as _sp
    import sys as _sys
    p = str(p)
    try:
        if _sys.platform == "darwin":
            _sp.run(["open", p], check=False)
        elif _sys.platform.startswith("win"):
            import os as _os
            _os.startfile(p)
        else:
            _sp.run(["xdg-open", p], check=False)
    except Exception:
        pass

def load_cfg() -> dict:
    try:
        return {**DEFAULTS, **json.loads(CFG.read_text("utf-8"))}
    except Exception:
        return dict(DEFAULTS)


def save_cfg(d: dict) -> None:
    try:
        CFG.parent.mkdir(parents=True, exist_ok=True)
        CFG.write_text(json.dumps(d, ensure_ascii=False, indent=2), "utf-8")
    except Exception:
        pass


# ---------- 许可信息（PWS 方法来源）----------
LICENSE_TEXT = (
    "分区加权叠加（PWS）+ Drizzle 几何重采样\n"
    "\n"
    "本软件的方法基于 D.Cikey 提出的分区权重叠加法\n"
    "（Partition-Weighted Stacking，PWS）。\n"
    "\n"
    "原文（GitHub）：\n"
    "https://github.com/DDCikey/\n"
    "Partition-Weighted-Stacking/tree/main/docs\n"
    "\n"
    "PDF 版可在微信公众号「小丁的星空」下载。\n"
    "\n"
    "软件作者：打野波比\n"
    "\n"
    "版权归原作者所有；本软件是在该方法公开描述基础上做的\n"
    "独立实现，并增加了配准、drizzle、GPU 加速、自动裁边、\n"
    "评价与调参等功能。"
)


def _argv_common(c) -> list:
    """按界面设置组叠加命令行；调参也走同一份，保证两边口径一致。"""
    argv = [str(PYEXE), str(MAINPY), "--cli",
            "--photos", c["photos"], "--out", c["out"]]
    if c["tag"]:
        argv += ["--tag", c["tag"]]
    for key, flag in (("A", "--A"), ("rej_k", "--rej-k"), ("n_star", "--n-star"),
                      ("limit", "--limit"), ("crop", "--crop")):
        if c[key]:
            argv += [flag, c[key]]
    if c["drizzle"]:
        argv += ["--drizzle", "--drizzle-scale", c["scale"] or "2",
                 "--drizzle-pix", c["pix"] or "1.0"]
    else:
        # 取消勾选 = 不做放大，但仍然配准：走 1× 网格的几何重采样
        argv += ["--drizzle", "--drizzle-scale", "1", "--drizzle-pix", "1.0"]
        if c.get("r_blur"):
            argv += ["--r-blur", c["r_blur"]]
        if c.get("xdrz"):
            argv += ["--xdrz", c["xdrz"]]
        if c.get("r_mode") and c["r_mode"] != "max":
            argv += ["--r-mode", c["r_mode"]]
    if c.get("pick"):
        argv += ["--pick", c["pick"]]
    if c.get("bgdeg") not in ("", None):
        argv += ["--bgdeg", c["bgdeg"]]
    if c.get("sharp_w"):
        argv += ["--sharp-w", c["sharp_w"]]
    if c.get("fwhm_max"):
        argv += ["--fwhm-max", c["fwhm_max"]]
    if c.get("flux_pow"):
        argv += ["--flux-pow", c["flux_pow"]]
    if c.get("crop_cover"):
        argv += ["--crop-cover", c["crop_cover"]]
    if c.get("frame"):
        argv += ["--frame", c["frame"]]
    if c.get("match"):
        argv += ["--match", c["match"]]
    if c.get("gpu") and c["gpu"] != "auto":
        argv += ["--gpu", c["gpu"]]
    if str(c.get("aligned", "0")) == "1":
        argv += ["--aligned", "1"]
    if c.get("cfa") and c["cfa"] != "off":
        argv += ["--cfa", c["cfa"]]
    if c.get("cfa_debayer") and c["cfa_debayer"] != "bilinear":
        argv += ["--cfa-debayer", c["cfa_debayer"]]
    if c.get("cfa_weights") and c["cfa_weights"] != "shared":
        argv += ["--cfa-weights", c["cfa_weights"]]
    if str(c.get("native_1x", "0")) == "1":
        argv += ["--native-1x", "1"]
    return argv


class App(tk.Tk):
    def __init__(self, selftest: bool = False, smoke: bool = False):
        super().__init__()
        self.title("DWT · Drizzle")
        self.configure(bg=BG)
        self.selftest = selftest
        self.smoke = smoke

        self.cfg = load_cfg()
        self.proc: subprocess.Popen | None = None
        self.mon = None
        self._mon_buf = []          # 监控窗的日志/事件缓冲，窗口被关掉重开时回放
        self.q: queue.Queue = queue.Queue()
        self.t0 = 0.0
        self.reader: threading.Thread | None = None
        self._build_style()
        self._build_ui()
        self._load_values()
        self.after(100, self._pump)
        self.after(500, self._tick)
        if not (selftest or smoke):
            self.after(350, self.show_license)   # 启动即弹许可说明
        if selftest:
            self.withdraw()
            self.after(600, self._selftest_done)

    # ---------- 外观 ----------
    def _build_style(self):
        style = ttk.Style(self)
        # clam 是唯一能稳定吃下自定义底色的内置主题；vista 会把 background 忽略掉
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(".", font=FONT, background=BG)
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG)
        style.configure("TCheckbutton", background=BG, foreground=FG)
        style.configure("TRadiobutton", background=BG, foreground=FG)
        style.configure("TLabelframe", background=CARD, bordercolor=BORDER,
                        relief="solid", borderwidth=1)
        style.configure("TLabelframe.Label", background=BG, foreground=ACCENT_D,
                        font=FONT_B)
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(16, 7), font=FONT,
                        background="#e5e7eb", foreground="#374151")
        style.map("TNotebook.Tab",
                  background=[("selected", CARD)], foreground=[("selected", ACCENT_D)])
        style.configure("TButton", padding=(12, 6), background="#ffffff",
                        bordercolor=BORDER, focusthickness=1, font=FONT)
        style.map("TButton",
                  background=[("active", "#eef2f7"), ("pressed", "#e2e8f0")])
        style.configure("Run.TButton", font=FONT_B, padding=(18, 8),
                        background=ACCENT, foreground="#ffffff", bordercolor=ACCENT)
        style.map("Run.TButton",
                  background=[("active", ACCENT_D), ("pressed", ACCENT_D),
                              ("disabled", "#b9c6c4")],
                  foreground=[("disabled", "#f1f5f9")])
        style.configure("TEntry", padding=4, fieldbackground="#ffffff")
        style.configure("TSpinbox", padding=3, fieldbackground="#ffffff")
        style.configure("TCombobox", padding=3)
        style.configure("TScrollbar", background="#dbe2ea", troughcolor=BG,
                        bordercolor=BG, arrowcolor="#64748b")

    def _build_ui(self):
        # 顶栏
        top = tk.Frame(self, bg="#171e29", height=64)
        top.pack(fill="x")
        top.pack_propagate(False)
        tk.Label(top, text="DWT", bg="#171e29", fg="#2dd4bf",
                 font=(_FONT_FAMILY, 18, "bold")).pack(side="left", padx=(20, 8))
        tk.Label(top, text="分区加权叠加  +  Drizzle 几何重采样", bg="#171e29",
                 fg="#cbd5e1", font=(_FONT_FAMILY, 11)).pack(side="left", pady=22)
        self.lab_state = tk.Label(top, text="空闲", bg="#171e29", fg="#94a3b8", font=FONT)
        self.lab_state.pack(side="right", padx=20)

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=10, pady=(8, 10))
        page1 = tk.Frame(self.nb, bg=BG)
        self.nb.add(page1, text="   叠加   ")

        # 小屏幕/高 DPI 下内容放不下 → 整页可滚动（滚轮 + 右侧滚动条）
        shell = tk.Frame(page1, bg=BG)
        shell.pack(fill="both", expand=True)
        self.cv_body = tk.Canvas(shell, bg=BG, highlightthickness=0)
        sb_body = ttk.Scrollbar(shell, orient="vertical", command=self.cv_body.yview)
        self.cv_body.configure(yscrollcommand=sb_body.set)
        self.cv_body.pack(side="left", fill="both", expand=True)
        sb_body.pack(side="right", fill="y")

        body = tk.Frame(self.cv_body, bg=BG)
        self._body_win = self.cv_body.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda e: self.cv_body.configure(
            scrollregion=self.cv_body.bbox("all")))
        self.cv_body.bind("<Configure>", lambda e: self.cv_body.itemconfigure(
            self._body_win, width=e.width))
        for w in (self.cv_body, body):
            w.bind("<MouseWheel>", self._wheel)
        body.columnconfigure(0, weight=1)

        # 路径
        p = ttk.Labelframe(body, text=" 素材与输出 ", padding=10)
        p.grid(row=0, column=0, sticky="ew")
        p.columnconfigure(1, weight=1)

        ttk.Label(p, text="素材目录").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.v_photos = tk.StringVar()
        e = ttk.Entry(p, textvariable=self.v_photos)
        e.grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Button(p, text="浏览…", width=8,
                   command=self._pick_photos).grid(row=0, column=2, padx=(8, 0))

        ttk.Label(p, text="输出目录").grid(row=1, column=0, sticky="w", padx=(0, 8))
        self.v_out = tk.StringVar()
        ttk.Entry(p, textvariable=self.v_out).grid(row=1, column=1, sticky="ew", pady=3)
        f2 = ttk.Frame(p)
        f2.grid(row=1, column=2, padx=(8, 0))
        ttk.Button(f2, text="浏览…", width=6,
                   command=self._pick_out).pack(side="left")
        ttk.Button(f2, text="打开", width=4,
                   command=self._open_out).pack(side="left", padx=(4, 0))

        ttk.Label(p, text="成品名").grid(row=2, column=0, sticky="w", padx=(0, 8))
        f3 = ttk.Frame(p)
        f3.grid(row=2, column=1, columnspan=2, sticky="w", pady=(4, 0))
        self.v_tag = tk.StringVar()
        ttk.Entry(f3, textvariable=self.v_tag, width=24).pack(side="left")
        ttk.Label(f3, text="   产物：PWS_<成品名>_<滤镜>.xisf（另附 R 场 / N_eff / 权重表）",
                  foreground="#6b7280", font=FONT_S).pack(side="left")

        ttk.Label(p, text="素材要「已校准、未对齐」：WBPP 的 calibrated\\ 可直接用，registered\\ 不要用；Flat / Dark 会自动跳过，各通道自动分组",
                  foreground="#6b7280", font=FONT_S).grid(row=3, column=0, columnspan=3,
                                                         sticky="w", pady=(2, 0))

        bar2 = tk.Frame(p)
        bar2.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        ttk.Button(bar2, text="扫描通道", width=10,
                   command=self._scan_channels).pack(side="left")
        ttk.Button(bar2, text="全选", width=5,
                   command=lambda: self._check_all(True)).pack(side="left", padx=(6, 0))
        ttk.Button(bar2, text="全不选", width=6,
                   command=lambda: self._check_all(False)).pack(side="left", padx=(4, 0))
        self.lab_ch = tk.Label(bar2, text="先扫描，再在下面选要叠的通道（不选 = 全部）",
                               bg=BG, fg="#6b7280", font=FONT_S)
        self.lab_ch.pack(side="left", padx=10)

        fr_lb = tk.Frame(p)
        fr_lb.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        self.cv_ch = tk.Canvas(fr_lb, height=108, highlightthickness=0, bg=BG)
        sb = ttk.Scrollbar(fr_lb, orient="vertical", command=self.cv_ch.yview)
        self.cv_ch.configure(yscrollcommand=sb.set)
        self.cv_ch.pack(side="left", fill="x", expand=True)
        sb.pack(side="right", fill="y")
        self.fr_ch = tk.Frame(self.cv_ch, bg=BG)
        self.ch_win = self.cv_ch.create_window((0, 0), window=self.fr_ch, anchor="nw")
        self.fr_ch.bind("<Configure>",
                        lambda e: self.cv_ch.configure(scrollregion=self.cv_ch.bbox("all")))
        self.cv_ch.bind("<Configure>",
                        lambda e: self.cv_ch.itemconfigure(self.ch_win, width=e.width))
        self.ch_names: list = []
        self.ch_vars: list = []

        # 参数
        g = ttk.Labelframe(body, text=" 叠加参数 ", padding=10)
        g.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.v_A = tk.StringVar()
        self.v_rejk = tk.StringVar()
        self.v_nstar = tk.StringVar()
        self.v_limit = tk.StringVar()
        self.v_crop = tk.StringVar()

        def cell(parent, col, label, var, width, tip):
            ttk.Label(parent, text=label).grid(row=0, column=col * 2, sticky="w",
                                               padx=(0 if col == 0 else 18, 6))
            ttk.Entry(parent, textvariable=var, width=width).grid(row=0, column=col * 2 + 1, sticky="w")
            ttk.Label(parent, text=tip, foreground="#9ca3af", font=FONT_S).grid(
                row=1, column=col * 2, columnspan=2, sticky="w")

        cell(g, 0, "清晰度 A", self.v_A, 8, "推荐 20（更锐）；14 更干净、有效帧更多")
        cell(g, 1, "排异 κ", self.v_rejk, 8, "3.5σ，越大越保守")
        cell(g, 2, "星点数", self.v_nstar, 8, "120，视场大可以加")
        cell(g, 3, "限帧", self.v_limit, 8, "只用前 N 帧；0 = 全部")

        # Drizzle
        d = ttk.Labelframe(body, text=" Drizzle ", padding=10)
        d.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.v_drz = tk.BooleanVar()
        self.v_scale = tk.StringVar()
        self.v_pix = tk.StringVar()
        self.v_rb = tk.StringVar()
        ttk.Checkbutton(d, text="启用 drizzle 放大（取消 = 1× 网格，仍做配准）",
                        variable=self.v_drz, command=self._sync_drz).grid(
            row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(d, text="输出倍率").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        self.sp_scale = ttk.Spinbox(d, from_=1.0, to=3.0, increment=0.5, width=8,
                                    textvariable=self.v_scale)
        self.sp_scale.grid(row=1, column=1, sticky="w", pady=(8, 0))
        ttk.Label(d, text="σ 1.2–2.5px 用 2.0；比 2.5 大用 1.5", foreground="#9ca3af",
                  font=FONT_S).grid(row=1, column=2, sticky="w", padx=12, pady=(8, 0))

        ttk.Label(d, text="pixfrac").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(4, 0))
        self.sp_pix = ttk.Spinbox(d, from_=0.3, to=1.0, increment=0.1, width=8,
                                  textvariable=self.v_pix)
        self.sp_pix.grid(row=2, column=1, sticky="w", pady=(4, 0))
        ttk.Label(d, text="1.0 最稳；确认有亚像素抖动才降到 0.7–0.8", foreground="#9ca3af",
                  font=FONT_S).grid(row=2, column=2, sticky="w", padx=12, pady=(4, 0))

        ttk.Label(d, text="裁剪边长").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=(4, 0))
        ttk.Entry(d, textvariable=self.v_crop, width=8).grid(row=3, column=1, sticky="w", pady=(4, 0))
        ttk.Label(d, text="0 = 整幅；例 500", foreground="#9ca3af",
                  font=FONT_S).grid(row=3, column=2, sticky="w", padx=12, pady=(4, 0))

        ttk.Label(d, text="R 场平滑").grid(row=4, column=0, sticky="w", padx=(0, 8), pady=(4, 0))
        ttk.Spinbox(d, from_=0.0, to=20.0, increment=1.0, width=8,
                    textvariable=self.v_rb).grid(row=4, column=1, sticky="w", pady=(4, 0))
        ttk.Label(d, text="0=原式；8 让星云区也参与锐度加权（本机实测最锐）", foreground="#9ca3af",
                  font=FONT_S).grid(row=4, column=2, sticky="w", padx=12, pady=(4, 0))


        # 高级
        adv = ttk.Labelframe(body, text=" 高级 ", padding=10)
        adv.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        adv.columnconfigure(1, weight=1)
        self.v_bgdeg = tk.StringVar()
        self.v_rmode = tk.StringVar()
        self.v_shw = tk.StringVar()
        self.v_fwhm = tk.StringVar()
        self.v_xdrz = tk.StringVar()
        self.v_match = tk.StringVar()
        self.v_flux = tk.StringVar()
        self.v_ccrop = tk.StringVar()
        self.v_frame = tk.StringVar()
        self.v_gpu = tk.StringVar()

        def arow(r, label, widget, tip):
            ttk.Label(adv, text=label).grid(row=r, column=0, sticky="w", padx=(0, 8),
                                            pady=(0 if r == 0 else 6, 0))
            widget.grid(row=r, column=1, sticky="ew" if r >= 4 else "w",
                        pady=(0 if r == 0 else 6, 0))
            ttk.Label(adv, text=tip, foreground="#9ca3af", font=FONT_S).grid(
                row=r, column=2, sticky="w", padx=12, pady=(0 if r == 0 else 6, 0))

        sp_bg = ttk.Spinbox(adv, from_=0, to=2, increment=1, width=6, textvariable=self.v_bgdeg)
        arow(0, "背景梯度", sp_bg, "0 关 / 1 平面（推荐）/ 2 二阶")
        cb_rm = ttk.Combobox(adv, values=["max", "band", "ctn"], state="readonly",
                             width=8, textvariable=self.v_rmode)
        arow(1, "R 场定义", cb_rm, "max 原式 / band 带通（星云细节好）/ ctn 对比度")
        arow(2, "清晰度范围", ttk.Entry(adv, textvariable=self.v_shw, width=8),
             "空/0 = 全场；2 = 只作用星点")
        arow(3, "FWHM 上限", ttk.Entry(adv, textvariable=self.v_fwhm, width=8),
             "0/空 = 不筛（一般用 A 就够）")
        fr_x = ttk.Frame(adv)
        ttk.Entry(fr_x, textvariable=self.v_xdrz).pack(side="left", fill="x", expand=True)
        ttk.Button(fr_x, text="浏览…", width=8, command=self._pick_xdrz).pack(side="left", padx=(8, 0))
        arow(4, "几何目录", fr_x, "WBPP 的 .xdrz 目录（跨夜旋转建议填）")
        fr_m = ttk.Frame(adv)
        ttk.Entry(fr_m, textvariable=self.v_match).pack(side="left", fill="x", expand=True)
        ttk.Button(fr_m, text="浏览…", width=8, command=self._pick_match).pack(side="left", padx=(8, 0))
        arow(5, "裁切参考", fr_m, "可选：叠完另存 _match.xisf")
        sp_fl = ttk.Spinbox(adv, from_=0, to=4, increment=0.5, width=6,
                            textvariable=self.v_flux)
        arow(6, "吞吐权重", sp_fl, "0 关 / 2 推荐：按每帧收光量加权（对齐 PSF 信号权重）")
        sp_cc = ttk.Spinbox(adv, from_=0.0, to=1.0, increment=0.05, width=6,
                            textvariable=self.v_ccrop)
        arow(7, "自动裁边", sp_cc,
             "按覆盖率裁掉空四角与薄边带，另存 _autocrop.xisf（0 关 / 0.9 推荐）")
        self.v_bgr = tk.StringVar(value=str(DEFAULTS.get("bg_r", "0")))
        sp_bgr = ttk.Spinbox(adv, from_=0, to=4, increment=2, width=6, textvariable=self.v_bgr)
        arow(8, "径向背景", sp_bgr,
             "0 关 / 2 拟 r² / 4 拟 r²+r⁴（抓渐晕与平场残差）")
        fr_fr = ttk.Frame(adv)
        ttk.Entry(fr_fr, textvariable=self.v_frame).pack(side="left", fill="x", expand=True)
        ttk.Button(fr_fr, text="浏览…", width=8,
                   command=self._pick_frame).pack(side="left", padx=(8, 0))
        arow(9, "输出网格", fr_fr,
             "填 master：成品宽高/朝向与它一致（不用再事后裁切）")
        cb_gpu = ttk.Combobox(adv, values=["自动", "只用 CPU", "只用 GPU"],
                              state="readonly", width=9, textvariable=self.v_gpu)
        arow(10, "计算设备", cb_gpu,
             "自动：有 NVIDIA 显卡就走 GPU；结果与 CPU 版逐项一致")
        self.v_tunea = tk.StringVar()
        arow(11, "调参 A 候选", ttk.Entry(adv, textvariable=self.v_tunea, width=18),
             "逗号分隔，默认 10,14,20；按暗部优先可写 6,10,14,20")
        self.v_1x = tk.BooleanVar(value=False)
        arow(12, "原尺寸副本", ttk.Checkbutton(adv, text="另存一份原生尺寸（1×，块平均）",
                                          variable=self.v_1x),
             "叠加仍在 N× 网格上做；做完块平均缩回原生像素，另存 _1x.xisf")
        self.v_aligned = tk.BooleanVar(value=False)
        arow(13, "对齐输入", ttk.Checkbutton(
            adv, text="输入已对齐（跳过配准与插值）", variable=self.v_aligned),
             "素材是 WBPP registered 目录时打开：不配准不插值，更快更锐（实测 2.47 vs 2.72）")
        self.v_cfa = tk.StringVar()
        arow(14, "彩机拜耳阵列", ttk.Combobox(
            adv, state="readonly", width=12, textvariable=self.v_cfa,
            values=["关", "自动识别", "RGGB", "BGGR", "GRBG", "GBRG"]),
             "彩机（OSC）原始帧：选「自动识别」读帧头；认不出会报错让你手动选")
        self.v_cfad = tk.StringVar()
        arow(15, "拆色算法", ttk.Combobox(
            adv, state="readonly", width=12, textvariable=self.v_cfad,
            values=["双线性", "2×2 合并"]),
             "双线性＝全分辨率（推荐）；2×2 合并＝颜色最准但宽高各减半")
        self.v_cfaw = tk.StringVar()
        arow(16, "彩色权重", ttk.Combobox(
            adv, state="readonly", width=12, textvariable=self.v_cfaw,
            values=["共用", "逐通道"]),
             "共用＝三通道用同一套 C/S/T（与 PI 一致、更快）；逐通道＝各算各的")

        # 操作
        bar = tk.Frame(body, bg=BG)
        bar.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        self.btn_run = ttk.Button(bar, text="▶  开始叠加", style="Run.TButton",
                                  command=self.start)
        self.btn_run.pack(side="left")
        self.btn_stop = ttk.Button(bar, text="■ 停止", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left")
        ttk.Button(bar, text="存为默认", width=10, command=self.save_default).pack(
            side="left", padx=(16, 0))
        ttk.Button(bar, text="查看权重场", width=12,
                   command=self.show_maps).pack(side="left", padx=(8, 0))
        ttk.Button(bar, text="运行监控", width=10,
                   command=self.show_monitor).pack(side="left", padx=(8, 0))
        ttk.Button(bar, text="使用说明", width=10,
                   command=self.show_help).pack(side="left", padx=(8, 0))
        ttk.Button(bar, text="关于", width=6,
                   command=self.show_about).pack(side="left", padx=(8, 0))
        ttk.Button(bar, text="自动调参", width=10,
                   command=self.start_tune).pack(side="left", padx=(8, 0))
        self.pb = ttk.Progressbar(bar, mode="determinate", value=0, length=180)
        self.pb.pack(side="left", padx=16)
        self.lab_time = tk.Label(bar, text="用时 00:00", bg=BG, fg="#4b5563", font=FONT)
        self.lab_time.pack(side="right")
        self.lab_stage = tk.Label(bar, text="阶段：—", bg=BG, fg="#4b5563", font=FONT)
        self.lab_stage.pack(side="right", padx=16)

        # 日志
        lg = ttk.Labelframe(body, text=" 日志 ", padding=6)
        lg.grid(row=5, column=0, sticky="nsew", pady=(10, 0))
        body.rowconfigure(5, weight=1)
        self.txt = tk.Text(lg, height=16, bg="#0f172a", fg="#cbd5e1", insertbackground="#cbd5e1",
                           font=("Consolas", 9), relief="flat", wrap="none", padx=8, pady=6)
        vs = ttk.Scrollbar(lg, orient="vertical", command=self.txt.yview)
        self.txt.configure(yscrollcommand=vs.set)
        self.txt.pack(side="left", fill="both", expand=True)
        vs.pack(side="right", fill="y")
        self.txt.tag_configure("err", foreground="#fca5a5")
        self.txt.tag_configure("warn", foreground="#fcd34d")
        self.txt.tag_configure("ok", foreground="#86efac")
        self.txt.tag_configure("dim", foreground="#64748b")
        if EvalPage is None:
            fr = tk.Frame(self.nb, bg=BG)
            tk.Label(fr, text="叠加评价页没加载起来：" + str(_EVAL_ERR) +
                     "（主界面的叠加功能不受影响）",
                     bg=BG, fg="#b45309", font=FONT, justify="left",
                     padx=20, pady=20).pack(anchor="w")
            self.nb.add(fr, text="   叠加评价   ")
            self.page_eval = None
        else:
            try:
                self.page_eval = EvalPage(self.nb, self)
                self.nb.add(self.page_eval, text="   叠加评价   ")
            except Exception as _e:      # 评价页出问题也不要把主界面拖死
                self.page_eval = None
                print("评价页加载失败：", _e)

        if MapsPage is None:
            fr2 = tk.Frame(self.nb, bg=BG)
            tk.Label(fr2, text="权重场页没加载起来：" + str(_MAPS_ERR),
                     bg=BG, fg="#b45309", font=FONT, justify="left",
                     padx=20, pady=20).pack(anchor="w")
            self.nb.add(fr2, text="   权重场   ")
            self.page_maps = None
        else:
            try:
                self.page_maps = MapsPage(self.nb, self)
                self.nb.add(self.page_maps, text="   权重场   ")
            except Exception:
                self.page_maps = None

        self.page_detect = None

        self._fit_window()

    def _fit_window(self):
        """按屏幕可用区自适应：小屏不超屏、大屏给足，记住上次大小。"""
        self.update_idletasks()
        sc = max(1.0, self.winfo_fpixels("1i") / 96.0)
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        avail_w, avail_h = sw - 60, sh - 100
        w = min(int(1040 * sc), avail_w)
        h = min(int(940 * sc), avail_h)
        w = max(w, min(int(860 * sc), avail_w))
        h = max(h, min(int(600 * sc), avail_h))
        geo = str(self.cfg.get("geometry", "") or "")
        if re.fullmatch(r"\d+x\d+", geo or ""):
            gw, gh = (int(v) for v in geo.split("x"))
            w = max(min(gw, avail_w), min(int(720 * sc), avail_w))
            h = max(min(gh, avail_h), min(int(520 * sc), avail_h))
        self.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}")
        self.minsize(min(int(720 * sc), avail_w), min(int(520 * sc), avail_h))

    def _wheel(self, ev):
        try:
            self.cv_body.yview_scroll(int(-1 * (ev.delta / 120.0)), "units")
        except Exception:
            pass

    def show_maps(self):
        if self.page_maps is None:
            messagebox.showinfo("权重场", "这个包里的权重场页没加载起来。")
            return
        self.nb.select(self.page_maps)
        self.page_maps.load_last()

    # ---------- 取值 / 存值 ----------
    def _load_values(self):
        c = self.cfg
        self.v_photos.set(c["photos"])
        self.v_out.set(c["out"])
        self.v_tag.set(c["tag"])
        self.v_A.set(c["A"])
        self.v_rejk.set(c["rej_k"])
        self.v_nstar.set(c["n_star"])
        self.v_limit.set(c["limit"])
        self.v_crop.set(c["crop"])
        self.v_drz.set(bool(c["drizzle"]))
        self.v_scale.set(c["scale"])
        self.v_pix.set(c["pix"])
        self.v_rb.set(c.get("r_blur", ""))
        self.v_bgdeg.set(str(c.get("bgdeg", "1")))
        self.v_bgr.set(str(c.get("bg_r", "0")))
        self.v_rmode.set(c.get("r_mode", "max"))
        self.v_shw.set(c.get("sharp_w", ""))
        self.v_fwhm.set(c.get("fwhm_max", ""))
        self.v_flux.set(c.get("flux_pow", "2.0"))
        self.v_ccrop.set(c.get("crop_cover", "0.9"))
        self.v_gpu.set({"auto": "自动", "cpu": "只用 CPU", "gpu": "只用 GPU"}.get(c.get("gpu", "auto"), "自动"))
        self.v_frame.set(c.get("frame", ""))
        self.v_tunea.set(c.get("tune_a", "10,14,20"))
        self.v_1x.set(str(c.get("native_1x", "0")) == "1")
        self.v_aligned.set(str(c.get("aligned", "0")) == "1")
        self.v_cfa.set({"off": "关", "auto": "自动识别"}.get(c.get("cfa", "off"),
                                                       c.get("cfa", "off") or "关"))
        self.v_cfad.set({"bilinear": "双线性", "superpixel": "2×2 合并"}.get(
            c.get("cfa_debayer", "bilinear"), "双线性"))
        self.v_cfaw.set({"shared": "共用", "own": "逐通道"}.get(
            c.get("cfa_weights", "shared"), "共用"))
        self.v_xdrz.set(c.get("xdrz", ""))
        self.v_match.set(c.get("match", ""))
        self._pick_saved = (c.get("pick") or "")
        self._sync_drz()

    def save_default(self):
        c = self._collect()
        c["geometry"] = "%dx%d" % (self.winfo_width(), self.winfo_height())
        save_cfg(c)
        self.lab_state.configure(text="已存默认", fg="#34d399")
        messagebox.showinfo("已保存", "当前这套参数已存为默认，下次启动自动带出来。")

    def _collect(self) -> dict:
        return {
            "photos": self.v_photos.get().strip(),
            "out": self.v_out.get().strip(),
            "tag": self.v_tag.get().strip(),
            "A": self.v_A.get().strip(),
            "rej_k": self.v_rejk.get().strip(),
            "n_star": self.v_nstar.get().strip(),
            "limit": self.v_limit.get().strip(),
            "crop": self.v_crop.get().strip(),
            "drizzle": bool(self.v_drz.get()),
            "scale": self.v_scale.get().strip(),
            "pix": self.v_pix.get().strip(),
            "r_blur": self.v_rb.get().strip(),
            "bgdeg": self.v_bgdeg.get().strip(),
            "r_mode": self.v_rmode.get().strip(),
            "sharp_w": self.v_shw.get().strip(),
            "fwhm_max": self.v_fwhm.get().strip(),
            "flux_pow": self.v_flux.get().strip(),
            "crop_cover": self.v_ccrop.get().strip(),
            "gpu": {"自动": "auto", "只用 CPU": "cpu", "只用 GPU": "gpu"}.get(self.v_gpu.get().strip(), "auto"),
            "frame": self.v_frame.get().strip(),
            "tune_a": self.v_tunea.get().strip() or "10,14,20",
            "native_1x": "1" if bool(self.v_1x.get()) else "0",
            "aligned": "1" if bool(self.v_aligned.get()) else "0",
            "cfa": {"关": "off", "自动识别": "auto"}.get(self.v_cfa.get().strip(),
                                                        self.v_cfa.get().strip() or "off"),
            "cfa_debayer": {"双线性": "bilinear", "2×2 合并": "superpixel"}.get(
                self.v_cfad.get().strip(), "bilinear"),
            "cfa_weights": {"共用": "shared", "逐通道": "own"}.get(
                self.v_cfaw.get().strip(), "shared"),
            "xdrz": self.v_xdrz.get().strip(),
            "match": self.v_match.get().strip(),
            "pick": self._pick_value(),
        }

    def _sync_drz(self):
        st = "normal" if self.v_drz.get() else "disabled"
        self.sp_scale.configure(state=st)
        self.sp_pix.configure(state=st)

    # ---------- 按钮 ----------
    def _pick_photos(self):
        d = filedialog.askdirectory(title="选择素材目录（已校准的 XISF / FITS，未对齐最好）",
                                    initialdir=self.v_photos.get() or str(HERE))
        if d:
            self.v_photos.set(d)

    def _pick_out(self):
        d = filedialog.askdirectory(title="选择输出目录",
                                    initialdir=self.v_out.get() or str(HERE))
        if d:
            self.v_out.set(d)

    def _scan_channels(self):
        """读帧头列出通道（不含像素），勾选后 --pick 只叠所选"""
        photos = self.v_photos.get().strip()
        if not photos or not Path(photos).exists():
            messagebox.showerror("素材不存在", "请先在「素材目录」里选一个存在的目录。")
            return
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(HERE / "pydeps"), str(HERE), str(HERE / "core")])
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            r = subprocess.run([str(PYEXE), str(MAINPY), "--cli", "--scan",
                                "--photos", photos],
                               cwd=str(HERE), env=env, capture_output=True,
                               text=True, encoding="utf-8", errors="replace",
                               timeout=900)
        except Exception as e:
            messagebox.showerror("扫描失败", str(e))
            return
        info = None
        for line in reversed((r.stdout or "").splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    info = json.loads(line)
                    break
                except Exception:
                    continue
        if not info:
            messagebox.showerror("扫描失败",
                                 ((r.stdout or "") + (r.stderr or ""))[-1200:])
            return
        self._ch_total = int(info.get("total", 0) or 0)
        self.ch_names = []
        self.ch_vars = []
        for w in self.fr_ch.winfo_children():
            w.destroy()
        _want = set(x for x in (getattr(self, "_pick_saved", "") or "").split(";") if x)
        for c in info.get("channels", []):
            self.ch_names.append(c["name"])
            v = tk.BooleanVar(value=(not _want) or (c["name"] in _want))
            self.ch_vars.append(v)
            ttk.Checkbutton(self.fr_ch, text="☑ %s   ·   %d 帧" % (c["label"], c["n"]),
                            variable=v,
                            command=self._sync_check_glyph).pack(anchor="w", padx=2)
        self._sync_check_glyph()
        self.fr_ch.update_idletasks()
        self.cv_ch.configure(scrollregion=self.cv_ch.bbox("all"))
        self._sync_check_glyph()

    def _check_all(self, flag: bool):
        for v in getattr(self, "ch_vars", []):
            v.set(bool(flag))
        self._sync_check_glyph()

    def _sync_check_glyph(self):
        """☑ / ☐ 跟着状态走，一眼看出勾了哪些"""
        if not getattr(self, "ch_vars", None):
            return
        for w, v in zip(self.fr_ch.winfo_children(), self.ch_vars):
            try:
                txt = w.cget("text")
            except Exception:
                continue
            if txt[:1] in ("☑", "☐"):
                w.configure(text=("☑" if v.get() else "☐") + txt[1:])
        n_on = sum(1 for v in self.ch_vars if v.get())
        tot = getattr(self, "_ch_total", 0)
        head = ("共 %d 帧 → %d 个通道" % (tot, len(self.ch_vars))) if tot else                ("%d 个通道" % len(self.ch_vars))
        tail = ("，全选 = 不加筛选" if n_on == len(self.ch_vars) else
                ("，一个都没选也按全部处理" if n_on == 0 else "，只叠勾选的"))
        self.lab_ch.configure(text=head + "，勾选 %d 个%s" % (n_on, tail))
        self._pick_saved = self._pick_value()

    def _pick_value(self) -> str:
        if not getattr(self, "ch_names", None):
            return getattr(self, "_pick_saved", "") or ""
        sel = [n for n, v in zip(self.ch_names, self.ch_vars) if v.get()]
        if not sel or len(sel) == len(self.ch_names):
            return ""
        return ";".join(x if x else "无滤镜信息" for x in sel)

    def _pick_frame(self):
        p = filedialog.askopenfilename(
            title="选一张参考图（master）", filetypes=[("XISF", "*.xisf"), ("全部", "*.*")])
        if p:
            self.v_frame.set(p)

    def _pick_match(self):
        f = filedialog.askopenfilename(
            title="选择参考图（master .xisf）",
            initialdir=str(Path(self.v_match.get()).parent) if self.v_match.get() else str(HERE),
            filetypes=[("XISF", "*.xisf"), ("所有文件", "*.*")])
        if f:
            self.v_match.set(f)

    def _pick_xdrz(self):
        d = filedialog.askdirectory(title="选择 .xdrz 所在目录（WBPP registered\\ 下）",
                                    initialdir=self.v_xdrz.get() or str(HERE))
        if d:
            self.v_xdrz.set(d)

    def _open_out(self):
        p = self.v_out.get().strip()
        if p and Path(p).exists():
            _open_path(p)
        else:
            messagebox.showinfo("提示", "输出目录还不存在。")

    # ---------- 启动时的许可提示 ----------
    def show_license(self):
        """开机弹窗：PWS 方法来源与版权说明。点「我知道了」才能继续。"""
        if os.environ.get("DWT_NO_STARTUP_DIALOG") == "1":
            return
        win = tk.Toplevel(self)
        win.title("关于本软件的方法来源")
        win.configure(bg=CARD)
        win.transient(self)
        win.resizable(False, False)
        try:
            win.grab_set()
        except Exception:
            pass

        head = tk.Frame(win, bg=CARD)
        head.pack(fill="x", padx=22, pady=(18, 6))
        tk.Label(head, text="DWT · Drizzle", bg=CARD, fg=ACCENT,
                 font=(_FONT_FAMILY, 15, "bold")).pack(anchor="w")
        tk.Label(head, text="使用前请先阅读下面的方法来源与版权说明",
                 bg=CARD, fg="#64748b", font=(_FONT_FAMILY, 10)).pack(anchor="w", pady=(4, 0))

        body = tk.Text(win, wrap="word", font=(_FONT_FAMILY, 10), bg="#f8fafc", fg=FG,
                       relief="flat", padx=14, pady=12, width=64, height=16,
                       highlightthickness=1, highlightbackground=BORDER)
        body.pack(fill="both", expand=True, padx=22, pady=(6, 12))
        body.insert("1.0", LICENSE_TEXT)
        body.configure(state="disabled")

        bar = tk.Frame(win, bg=CARD)
        bar.pack(fill="x", padx=22, pady=(0, 18))
        ttk.Button(bar, text="打开原文链接",
                   command=lambda: _open_path("https://github.com/DDCikey/"
                                              "Partition-Weighted-Stacking/tree/main/docs")
                   ).pack(side="left")
        ok = ttk.Button(bar, text="我知道了", style="Run.TButton",
                        command=lambda: self._close_license(win))
        ok.pack(side="right")
        win.bind("<Return>", lambda e: self._close_license(win))
        win.bind("<Escape>", lambda e: self._close_license(win))

        win.update_idletasks()
        w, h = win.winfo_reqwidth(), win.winfo_reqheight()
        sw = self.winfo_screenwidth() or 1600     # 未映射的 Toplevel 可能返回 0，用主窗的问
        sh = self.winfo_screenheight() or 900
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 3)
        win.geometry("%dx%d+%d+%d" % (w, h, x, y))
        # 再补一次：有些 WM 会在映射时把 transient 挪回 (0,0)
        win.after(60, lambda: win.winfo_exists() and win.geometry("+%d+%d" % (x, y)))
        win.attributes("-topmost", True)
        win.after(600, lambda: win.winfo_exists() and win.attributes("-topmost", False))
        win.lift()
        win.focus_force()
        win.protocol("WM_DELETE_WINDOW", lambda: self._close_license(win))
        ok.focus_set()
        self._license_win = win

    def _close_license(self, win):
        try:
            win.grab_release()
        except Exception:
            pass
        self._license_win = None
        win.destroy()

    def show_about(self):
        """关于：方法与原文出处（PWS，D.Cikey）"""
        messagebox.showinfo(
            "关于 DWT · Drizzle",
            "DWT · Drizzle 2.7\n\n" + LICENSE_TEXT)

    def show_help(self):
        """打开内置使用说明（不依赖浏览器 / 外部程序）"""
        if _hp is None:
            messagebox.showinfo("使用说明", "找不到使用说明文件")
            return None
        try:
            return _hp.open_help(self)
        except Exception as e:
            messagebox.showinfo("使用说明", "打不开内置说明：%s" % e)
            return None

    def show_monitor(self):
        """打开/唤起运行监控窗（叠加开始时会自动打开）"""
        if Monitor is None:
            messagebox.showinfo("运行监控", "监控模块不可用")
            return
        fresh = self.mon is None or not self.mon.winfo_exists()
        if fresh:
            self.mon = Monitor(self)
            try:                       # 窗口曾被杀掉时，把已有日志回放进去
                for ln in list(getattr(self, '_mon_buf', [])):
                    self.mon.feed(ln)
            except Exception:
                pass
        self.mon.deiconify()
        self.mon.lift()

    def start_tune(self):
        """叠加前自动调参：抽若干帧试几组 A / r_mode，给出建议。"""
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("正在运行", "已有任务在跑，等它结束或先停止。")
            return
        c = self._collect()
        if not c["photos"] or not Path(c["photos"]).exists():
            messagebox.showerror("素材不存在", f"素材目录不存在：{c['photos']}")
            return
        if not c["out"]:
            messagebox.showerror("缺少输出目录", "请先选择输出目录。")
            return
        n = simpledialog.askinteger("自动调参", "抽几帧试参数？（建议 16，越多越准也越慢）",
                                    initialvalue=16, minvalue=6, maxvalue=64, parent=self)
        if not n:
            return
        save_cfg(c)
        try:
            self.show_monitor()
            self.mon.reset()
            self._mon_buf = []
        except Exception:
            pass
        argv = _argv_common(c) + ["--tune", str(int(n))]
        if c.get("tune_a"):
            argv += ["--tune-a", c["tune_a"]]
        self._tune_mode = True
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(HERE / "pydeps"), str(HERE)])
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        self.txt.delete("1.0", "end")
        self._log("命令：" + " ".join(argv) + chr(10), "dim")
        try:
            self.proc = subprocess.Popen(
                argv, cwd=str(HERE), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        except Exception as e:
            messagebox.showerror("启动失败", str(e))
            return
        self.t0 = time.time()
        self.pb.configure(mode="indeterminate")
        self.pb.start(12)
        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.lab_state.configure(text="调参中", fg="#fbbf24")
        self.reader = threading.Thread(target=self._read, args=(self.proc,), daemon=True)
        self.reader.start()

    def _tune_best(self, line: str):
        """收到 TUNE-BEST 就把建议填回界面。"""
        try:
            import json as _json
            d = _json.loads(line.split('TUNE-BEST', 1)[1].strip())
        except Exception:
            return
        try:
            self.v_A.set('%g' % d['A'])
            self.v_rmode.set(d['r_mode'])
        except Exception:
            return
        msg = ('建议：清晰度 A = %g，R 场定义 = %s') % (d['A'], d['r_mode'])
        msg += '\n' + ('成品 FWHM %.2fpx，改善 %+.1f%%，N_eff %.1f' % (d['fwhm'], d['gain'], d.get('neff', 0)))
        msg += '\n\n已填回界面，直接点「开始叠加」即可。'
        messagebox.showinfo('自动调参结论', msg)

    def start(self):
        if self.proc and self.proc.poll() is None:
            return
        c = self._collect()
        if not c["photos"] or not Path(c["photos"]).exists():
            messagebox.showerror("素材不存在", f"素材目录不存在：\n{c['photos']}")
            return
        if not c["out"]:
            messagebox.showerror("缺少输出目录", "请先选择输出目录。")
            return
        save_cfg(c)
        try:
            self.show_monitor()
            self.mon.reset()
        except Exception:
            pass

        argv = _argv_common(c)


        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(HERE / "pydeps"), str(HERE)])
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        self.txt.delete("1.0", "end")
        self._log("命令：" + " ".join(argv) + "\n", "dim")
        try:
            self.proc = subprocess.Popen(
                argv, cwd=str(HERE), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        except Exception as e:
            messagebox.showerror("启动失败", str(e))
            return

        self.t0 = time.time()
        self.pb.configure(mode="indeterminate")
        self.pb.start(12)
        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.lab_state.configure(text="运行中", fg="#34d399")
        self.reader = threading.Thread(target=self._read, args=(self.proc,), daemon=True)
        self.reader.start()

    def stop(self):
        if not self.proc or self.proc.poll() is not None:
            return
        self._log("\n[界面] 正在停止…\n", "warn")
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                           capture_output=True)
        except Exception:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def _read(self, proc):
        try:
            for line in proc.stdout:      # type: ignore[union-attr]
                self.q.put(line.rstrip("\n"))
        except Exception as e:
            self.q.put(f"[界面] 读取日志出错：{e}")
        proc.wait()
        self.q.put(("__done__", proc.returncode))

    # ---------- 事件循环 ----------
    def _pump(self):
        try:
            for _ in range(200):
                item = self.q.get_nowait()
                if isinstance(item, tuple):
                    self._finish(item[1])
                else:
                    self._log_line(item)
        except queue.Empty:
            pass
        self.after(80, self._pump)

    def _log_line(self, line: str):
        if 'TUNE-BEST' in line:
            self._tune_best(line)
        tag = ""
        if ("Traceback" in line or "Error" in line or "错误" in line
                or "失败" in line and "匹配失败" not in line):
            tag = "err"
        elif "WARNING" in line or "注意" in line:
            tag = "warn"
        elif line.startswith("[时]") or "完成" in line:
            tag = "ok"
        self._log(line + "\n", tag)
        try:
            self._mon_buf.append(line)
            if len(self._mon_buf) > 4000:
                del self._mon_buf[:1000]
        except Exception:
            pass
        try:
            if self.mon is not None and self.mon.winfo_exists():
                self.mon.feed(line)
        except Exception:
            pass
        m = STAGE_RE.search(line)
        if m:
            stage = {"S": "信噪比权重", "σ": "噪声", "C": "测星点", "R 场": "R 场",
                     "叠加": "叠加", "输出": "输出", "位移": "配准", "星表": "星表",
                     "drizzle": "配准"}.get(m.group(1), m.group(1))
            self.lab_stage.configure(text=f"阶段：{stage}")

    def _log(self, s: str, tag: str = ""):
        at_bottom = self.txt.yview()[1] > 0.98
        self.txt.insert("end", s, tag)
        if at_bottom:
            self.txt.see("end")

    def _finish(self, code: int):
        try:
            if self.mon is not None and self.mon.winfo_exists():
                self.mon.finish(code)
        except Exception:
            pass
        self.pb.stop()
        self.pb.configure(mode="determinate", value=0)
        self.btn_run.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        if code == 0:
            self.lab_state.configure(text="完成", fg="#34d399")
            self._log("\n[界面] 完成。产物在：" + self.v_out.get() + "\n", "ok")
        else:
            self.lab_state.configure(text=f"失败（{code}）", fg="#f87171")
            self._log(f"\n[界面] 进程退出码 {code}。\n", "err")
        self.lab_stage.configure(text="阶段：—")
        if self.smoke:
            print("SMOKE-OK" if code == 0 else f"SMOKE-FAIL code={code}", flush=True)
            self.after(300, self.destroy)

    def _tick(self):
        if self.proc and self.proc.poll() is None and self.t0:
            s = int(time.time() - self.t0)
            self.lab_time.configure(text=f"用时 {s // 60:02d}:{s % 60:02d}")
        self.after(500, self._tick)

    def _selftest_done(self):
        self.destroy()
        print("SELFTEST-OK")

    def destroy(self):
        save_cfg(self._collect())
        if self.proc and self.proc.poll() is None:
            try:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                               capture_output=True)
            except Exception:
                pass
        super().destroy()


def _dpi_aware():
    """让窗口按物理像素绘制，避免高 DPI 下字体放大而布局被截断。"""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                import ctypes
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


def main():
    _dpi_aware()
    selftest = "--selftest" in sys.argv
    smoke = "--smoke" in sys.argv
    try:
        app = App(selftest=selftest, smoke=smoke)
    except Exception:
        import traceback
        msg = traceback.format_exc()
        try:
            Path(CFG.parent).mkdir(parents=True, exist_ok=True)
            (CFG.parent / "crash.log").write_text(msg, "utf-8")
        except Exception:
            pass
        try:
            r = tk.Tk()
            r.withdraw()
            messagebox.showerror("DWT 界面启动失败", msg)
            r.destroy()
        except Exception:
            pass
        raise
    if smoke:
        app.withdraw()
        app.v_limit.set("2")
        app.v_crop.set("400")
        app.v_drz.set(True)
        app.v_scale.set("2")
        app.after(400, app.start)
        app.after(300000, lambda: (print("SMOKE-TIMEOUT", flush=True), app.destroy()))
    app.mainloop()


if __name__ == "__main__":
    main()
