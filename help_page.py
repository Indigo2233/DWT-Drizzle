# -*- coding: utf-8 -*-
"""软件内置「使用说明」：把 使用说明.md 渲染成可滚动的只读窗口。

不引入 markdown 库：只做轻量处理——
  # / ## / ### 标题 → 加粗、放大、深色
  `代码` → 去掉反引号、换等宽字体色
  **粗体** → 去掉星号、加粗
  | 表格 | → 每行按列宽对齐成文本
  - 列表 → 保留缩进
"""
import re
import tkinter as tk
from tkinter import ttk
from pathlib import Path

import sys as _sys
_FONT_FAMILY = ("PingFang SC" if _sys.platform == "darwin"
                else "Microsoft YaHei UI" if _sys.platform.startswith("win")
                else "Noto Sans CJK SC")
FONT = (_FONT_FAMILY, 10)
FONT_S = (_FONT_FAMILY, 9)
FONT_B = (_FONT_FAMILY, 10, "bold")
FONT_H1 = (_FONT_FAMILY, 15, "bold")
FONT_H2 = (_FONT_FAMILY, 12, "bold")
MONO = ("Consolas", 9)
BG = "#eef1f5"
CARD = "#ffffff"
BORDER = "#d8dee6"
FG = "#1f2937"
DIM = "#6b7280"
ACCENT = "#0f766e"

_HERE = Path(__file__).resolve().parent
DOC_QUICK = _HERE / "快速上手.md"     # 内置说明默认显示这份（简版）
DOC_FULL = _HERE / "使用说明.md"      # 完整版，点「完整说明」切换
DOC = DOC_QUICK


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

def _clean(line: str) -> str:
    line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
    line = re.sub(r"`(.+?)`", r"\1", line)
    line = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1", line)
    return line


def _table_rows(lines, i):
    """把连续的 | a | b | 行转成对齐文本，返回 (文本块, 下一个下标)"""
    rows = []
    while i < len(lines) and lines[i].lstrip().startswith("|"):
        cells = [_clean(c.strip()) for c in lines[i].strip().strip("|").split("|")]
        if not all(set(c) <= set("-: ") for c in cells):
            rows.append(cells)
        i += 1
    if not rows:
        return "", i
    ncol = max(len(r) for r in rows)
    widths = [0] * ncol
    for r in rows:
        for k, c in enumerate(r):
            w = sum(2 if ord(ch) > 127 else 1 for ch in c)
            widths[k] = max(widths[k], w)

    def pad(s, w):
        cur = sum(2 if ord(ch) > 127 else 1 for ch in s)
        return s + " " * max(0, w - cur)

    out = []
    for r in rows:
        out.append("  ".join(pad(c, widths[k]) for k, c in enumerate(r)).rstrip())
    return "\n".join(out), i


def build_text(widget: tk.Text, md: str):
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    widget.tag_configure("h1", font=FONT_H1, foreground=ACCENT, spacing1=14, spacing3=6)
    widget.tag_configure("h2", font=FONT_H2, foreground=FG, spacing1=12, spacing3=4)
    widget.tag_configure("h3", font=FONT_B, foreground=FG, spacing1=8, spacing3=2)
    widget.tag_configure("body", font=FONT, foreground=FG, spacing1=2)
    widget.tag_configure("code", font=MONO, foreground="#0b5563", background="#f1f5f9")
    widget.tag_configure("dim", font=FONT_S, foreground=DIM)

    lines = md.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = _clean(raw).rstrip()
        if not line.strip():
            widget.insert("end", "\n")
            i += 1
            continue
        if line.lstrip().startswith("|"):
            block, i = _table_rows(lines, i)
            widget.insert("end", block + "\n\n", "code")
            continue
        if line.startswith("### "):
            widget.insert("end", line[4:] + "\n", "h3")
        elif line.startswith("## "):
            widget.insert("end", line[3:] + "\n", "h2")
        elif line.startswith("# "):
            widget.insert("end", line[2:] + "\n", "h1")
        elif line.startswith("---"):
            widget.insert("end", "─" * 46 + "\n", "dim")
        else:
            widget.insert("end", line + "\n", "body")
        i += 1
    widget.configure(state="disabled")


class HelpWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("使用说明")
        self.configure(bg=BG)
        w = min(900, max(640, self.winfo_screenwidth() // 2))
        h = min(820, max(520, self.winfo_screenheight() - 160))
        self.geometry("%dx%d+%d+%d" % (w, h, 80, 60))
        self.minsize(560, 420)
        self.cur = DOC_QUICK          # 默认档：简版（快速上手.md）

        head = tk.Frame(self, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        head.pack(fill="x", padx=12, pady=(12, 8))
        tk.Label(head, text="使用说明", bg=CARD, fg=ACCENT, font=FONT_H2).pack(
            side="left", padx=12, pady=8)
        self.lab_doc = tk.Label(head, text=self.cur.name, bg=CARD, fg=DIM, font=FONT_S)
        self.lab_doc.pack(side="left", padx=(0, 12))
        ttk.Button(head, text="用系统程序打开",
                   command=self._open_external).pack(side="right", padx=10, pady=6)
        self.btn_full = ttk.Button(head, text="完整说明",
                                   command=lambda: self._render(DOC_FULL))
        self.btn_full.pack(side="right", padx=(0, 6), pady=6)
        self.btn_quick = ttk.Button(head, text="简版",
                                    command=lambda: self._render(DOC_QUICK))
        self.btn_quick.pack(side="right", padx=(0, 6), pady=6)

        wrap = tk.Frame(self, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        wrap.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.txt = tk.Text(wrap, wrap="word", padx=16, pady=12, relief="flat",
                           bg=CARD, fg=FG, insertwidth=0, cursor="arrow")
        sb = ttk.Scrollbar(wrap, orient="vertical", command=self.txt.yview)
        self.txt.configure(yscrollcommand=sb.set)
        self.txt.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self._render(DOC_QUICK)
        self.txt.bind("<MouseWheel>", lambda e: self.txt.yview_scroll(-1 * int(e.delta / 120), "units"))

    # ---------- 渲染一份文档 ----------
    def _render(self, path):
        path = Path(path)
        self.cur = path
        try:
            md = path.read_text(encoding="utf-8")
        except Exception as e:      # 文档丢了也要能给句话
            md = "# 使用说明\n\n找不到 %s（%s）。\n\n看 README-drizzle.md 或 参数说明.md。" % (path, e)
        self.txt.configure(state="normal")
        self.txt.delete("1.0", "end")
        build_text(self.txt, md)
        self.txt.yview_moveto(0)
        try:
            self.lab_doc.configure(text=path.name)
            # 当前看哪份，就把另一份的按钮高亮出来
            if path == DOC_FULL:
                self.btn_full.state(["disabled"])
                self.btn_quick.state(["!disabled"])
            else:
                self.btn_quick.state(["disabled"])
                self.btn_full.state(["!disabled"])
        except Exception:
            pass

    def _open_external(self):
        try:
            _open_path(str(self.cur))     # 交给系统默认程序
        except Exception:
            pass


def open_help(app):
    for ch in app.winfo_children():
        if isinstance(ch, HelpWindow) and ch.winfo_exists():
            ch.deiconify()
            ch.lift()
            return ch
    return HelpWindow(app)
