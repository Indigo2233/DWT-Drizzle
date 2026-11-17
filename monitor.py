# -*- coding: utf-8 -*-
"""叠加进度监控：单独窗口——阶段、进度、用时、被排除的帧、实时日志、SNR 对比。

日志行协议（由 core/pws.py 打出）：
  [EXCLUDED] {"n_used": 240, "n_total": 267, "items": [...]}
  [drizzle] 分带 12/40
  [SNR] 单张 SNR 中位 45.2（噪声 3.1e-04）→ 成品 SNR 中位 610.4（噪声 2.1e-05）  提升 13.5×
"""
import json
import re
import time
import tkinter as tk
from tkinter import ttk

import sys as _sys
_FONT_FAMILY = ("PingFang SC" if _sys.platform == "darwin"
                else "Microsoft YaHei UI" if _sys.platform.startswith("win")
                else "Noto Sans CJK SC")
FONT = (_FONT_FAMILY, 10)
FONT_S = (_FONT_FAMILY, 9)
FONT_B = (_FONT_FAMILY, 10, "bold")
FONT_SNR = (_FONT_FAMILY, 12, "bold")
BG = "#eef1f5"
CARD = "#ffffff"
BORDER = "#d8dee6"
ACCENT = "#0f766e"
FG = "#1f2937"
DIM = "#6b7280"
WARN = "#b45309"
LOGBG = "#0f172a"

STAGE_RE = re.compile(r"^\[([^\]]{1,12})\]")
BAND_RE = re.compile(r"带\s*(\d+)\s*/\s*(\d+)")
REJ_RE = re.compile(r"排异剔除率\s*([\d.]+)%")
SNR_RE = re.compile(r"单帧天空噪声 ([\d.eE+-]+) → 成品 ([\d.eE+-]+)  噪声降低 ([\d.]+)×")
NEFF_RE2 = re.compile(r"√N_eff = ([\d.]+)×")

STAGES = ["读帧", "信噪比权重", "噪声", "测星点", "星表", "配准", "吞吐", "R 场",
          "叠加", "输出"]
MAX_LOG = 3000


class Monitor(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("运行监控")
        self.configure(bg=BG)
        self.geometry("880x680+200+80")
        self.minsize(700, 520)
        self.t0 = None          # 没开始跑就不计时：等第一条日志进来才起表
        self.t_end = None
        self.excluded = []
        self.n_used = None
        self.n_total = None
        self.snr = None
        self.sqrt_neff = None
        self._build()
        self.after(500, self._tick)

    # ---------- 界面 ----------
    def _build(self):
        top = tk.Frame(self, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        top.pack(fill="x", padx=12, pady=(12, 10))
        head = tk.Frame(top, bg=CARD)
        head.pack(fill="x", padx=12, pady=(10, 0))
        self.lab_stage = tk.Label(head, text="等待开始…", bg=CARD, fg=FG, font=FONT_B)
        self.lab_stage.pack(side="left")
        self.lab_time = tk.Label(head, text="用时 00:00", bg=CARD, fg=DIM, font=FONT_S)
        self.lab_time.pack(side="right")
        self.protocol("WM_DELETE_WINDOW", self.hide)

        self.bar = ttk.Progressbar(top, mode="determinate", maximum=100)
        self.bar.pack(fill="x", padx=12, pady=(6, 2))
        self.lab_sub = tk.Label(top, text="", bg=CARD, fg=DIM, font=FONT_S)
        self.lab_sub.pack(anchor="w", padx=12)
        self.lab_snr = tk.Label(top, text="", bg=CARD, fg=ACCENT, font=FONT_SNR)
        self.lab_snr.pack(anchor="w", padx=12, pady=(2, 10))

        mid = tk.LabelFrame(self, text=" 被排除的帧 ", bg=CARD, fg=FG, font=FONT,
                            bd=1, relief="solid", padx=8, pady=6)
        mid.pack(fill="both", expand=False, padx=12, pady=(0, 10))
        self.lab_sum = tk.Label(mid, text="还没有收到帧清单", bg=CARD, fg=FG, font=FONT_S)
        self.lab_sum.pack(anchor="w", pady=(0, 6))
        wrap = tk.Frame(mid, bg=CARD)
        wrap.pack(fill="both", expand=True)
        cols = ("idx", "file", "reason", "fwhm")
        self.tv = ttk.Treeview(wrap, columns=cols, show="headings", height=6)
        for c, t, w in (("idx", "#", 48), ("file", "文件", 340),
                        ("reason", "原因", 300), ("fwhm", "FWHM(px)", 84)):
            self.tv.heading(c, text=t)
            self.tv.column(c, width=w, anchor="w" if c in ("file", "reason") else "center")
        sb = ttk.Scrollbar(wrap, orient="vertical", command=self.tv.yview)
        self.tv.configure(yscrollcommand=sb.set)
        self.tv.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.tv.tag_configure("bad", foreground=WARN)

        low = tk.LabelFrame(self, text=" 运行日志 ", bg=CARD, fg=FG, font=FONT,
                            bd=1, relief="solid", padx=6, pady=6)
        low.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.txt = tk.Text(low, height=10, font=FONT_S, bg=LOGBG, fg="#cbd5e1",
                           insertbackground="#cbd5e1", relief="flat", wrap="none")
        sb2 = ttk.Scrollbar(low, orient="vertical", command=self.txt.yview)
        self.txt.configure(yscrollcommand=sb2.set)
        self.txt.pack(side="left", fill="both", expand=True)
        sb2.pack(side="right", fill="y")
        self.txt.tag_configure("warn", foreground="#f59e0b")
        self.txt.tag_configure("err", foreground="#f87171")
        self.txt.tag_configure("ok", foreground="#34d399")
        self.txt.tag_configure("snr", foreground="#5eead4")
        self.txt.configure(state="disabled")

    # ---------- 计时 ----------
    def _tick(self):
        try:
            if not self.winfo_exists():
                return
        except Exception:
            return
        if self.t0 is None:
            self.lab_time.configure(text="用时 00:00")
        else:
            t = (self.t_end or time.time()) - self.t0
            m, s = divmod(int(t), 60)
            h, m = divmod(m, 60)
            txt = ("用时 %02d:%02d:%02d" % (h, m, s)) if h else ("用时 %02d:%02d" % (m, s))
            self.lab_time.configure(text=txt)
        self.after(500, self._tick)

    # ---------- 收日志 ----------
    def hide(self):
        """关闭按钮 = 收起来。窗口对象与计时都保留，再点「运行监控」就能原样回来。"""
        self.withdraw()

    def reset(self):
        self.t0 = None          # 复位也不起表
        self.t_end = None
        self.excluded = []
        self.n_used = self.n_total = None
        self.snr = None
        self.tv.delete(*self.tv.get_children())
        self.lab_sum.configure(text="正在读帧…")
        self.lab_stage.configure(text="读帧")
        self.lab_sub.configure(text="")
        self.lab_snr.configure(text="")
        self.bar.configure(value=0)
        self.txt.configure(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.configure(state="disabled")

    def finish(self, ok=None):
        self.t_end = time.time()
        self.bar.configure(value=100)
        self.lab_stage.configure(text="完成" if ok in (0, None) else "结束（有错误）")
        tail = "用时已定格"
        if self.snr:
            tail += "；噪声降低 %s×" % self.snr[2]
        self.lab_sub.configure(text=tail)

    def _show_snr(self):
        if not self.snr:
            return
        txt = "噪声降低 %s×（等效 SNR 提升）" % self.snr[2]
        if self.sqrt_neff:
            txt += "    √N_eff = %s×" % self.sqrt_neff
        self.lab_snr.configure(text=txt)

    def feed(self, line: str):
        """收一行日志：解析进度/排除/SNR，并按类型上色写进日志窗。"""
        if self.t0 is None and line.strip():
            self.t0 = time.time()
        if line.startswith("[EXCLUDED]"):
            # 机器数据，不进日志窗，只填表格
            try:
                self._fill(json.loads(line[len("[EXCLUDED]"):].strip()))
            except Exception:
                pass
            return
        self._append(line)
        m = SNR_RE.search(line)
        if m:
            self.snr = (m.group(1), m.group(2), m.group(3))
            self._show_snr()
        m = NEFF_RE2.search(line)
        if m:
            self.sqrt_neff = m.group(1)
            self._show_snr()
        m = BAND_RE.search(line)
        if m and ("drizzle" in line or "叠加" in line or "分带" in line):
            i, n = int(m.group(1)), max(1, int(m.group(2)))
            self.bar.configure(value=100.0 * i / n)
            self.lab_sub.configure(text="叠加 %d/%d 条带" % (i, n))
            self.lab_stage.configure(text="叠加（drizzle + 排异）")
            return
        m = REJ_RE.search(line)
        if m:
            self.lab_sub.configure(text="排异剔除率 %s%%" % m.group(1))
        m = STAGE_RE.match(line)
        if m:
            tag = m.group(1)
            name = {"S": "信噪比权重", "σ": "噪声", "C": "测星点", "T": "吞吐",
                    "位移": "配准", "星表": "星表", "R 场": "R 场", "叠加": "叠加",
                    "输出": "输出", "裁切": "裁切", "网格": "输出网格",
                    "背景": "背景梯度", "筛选": "帧筛选"}.get(tag, tag)
            if self.t_end is None:
                self.lab_stage.configure(text=name)

    def _append(self, line: str):
        tag = ""
        if "Traceback" in line or "Error" in line or "错误" in line:
            tag = "err"
        elif "WARNING" in line or "注意" in line:
            tag = "warn"
        elif line.startswith("[SNR]"):
            tag = "snr"
        elif line.startswith("[时]") or "完成" in line:
            tag = "ok"
        try:
            self.txt.configure(state="normal")
            self.txt.insert("end", line + "\n", tag)
            n = int(self.txt.index("end-1c").split(".")[0])
            if n > MAX_LOG:
                self.txt.delete("1.0", "%d.0" % (n - MAX_LOG + 200))
            self.txt.see("end")
            self.txt.configure(state="disabled")
        except Exception:
            pass

    def _fill(self, data):
        self.n_used = data.get("n_used")
        self.n_total = data.get("n_total")
        items = data.get("items") or []
        self.excluded = items
        self.tv.delete(*self.tv.get_children())
        for r in items:
            self.tv.insert("", "end", tags=("bad",), values=(
                r.get("idx"), r.get("file", ""), r.get("reason", ""),
                "" if r.get("fwhm") is None else r.get("fwhm")))
        if self.n_total:
            pct = 100.0 * (len(items) / self.n_total)
            self.lab_sum.configure(
                text="参与叠加 %s / %s 帧   被排除 %d 帧（%.1f%%）%s"
                     % (self.n_used, self.n_total, len(items), pct,
                        "" if items else "  —— 没有被排除的帧"))
        else:
            self.lab_sum.configure(text="被排除 %d 帧" % len(items))
