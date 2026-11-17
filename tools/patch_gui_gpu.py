# -*- coding: utf-8 -*-
"""v2.3 GUI：计算设备下拉（自动/CPU/GPU）、监控窗完成回调、argv --gpu。"""
from pathlib import Path

ROOT = Path(r"G:\DWT\dist\DWT-Drizzle-1.2")
ROOT2 = Path(r"G:\DWT\DWT-drizzle")


def sub1(s, old, new, tag):
    if old not in s:
        raise SystemExit("锚点缺失：%s" % tag)
    if s.count(old) != 1:
        raise SystemExit("锚点不唯一：%s" % tag)
    return s.replace(old, new, 1)


for root in (ROOT, ROOT2):
    p = root / "gui.py"
    s = p.read_text(encoding="utf-8")
    if "v_gpu" in s:
        print("gui 已打过", p)
        continue

    s = sub1(s, '    "pick": "H",', '    "pick": "H",\n    "gpu": "auto",', "DEFAULTS")

    s = sub1(s, "        self.v_frame = tk.StringVar()",
             "        self.v_frame = tk.StringVar()\n        self.v_gpu = tk.StringVar()", "var")

    s = sub1(s, '        arow(7, "自动裁边", sp_cc,',
             '        cb_gpu = ttk.Combobox(adv, values=["自动", "只用 CPU", "只用 GPU"],\n'
             '                              state="readonly", width=9, textvariable=self.v_gpu)\n'
             '        arow(9, "计算设备", cb_gpu,\n'
             '             "自动：有 NVIDIA 显卡就走 GPU；结果与 CPU 版逐项一致")\n'
             '        arow(7, "自动裁边", sp_cc,', "row")

    s = sub1(s, '        self.v_ccrop.set(c.get("crop_cover", "0.9"))',
             '        self.v_ccrop.set(c.get("crop_cover", "0.9"))\n'
             '        self.v_gpu.set({"auto": "自动", "cpu": "只用 CPU", "gpu": "只用 GPU"}'
             '.get(c.get("gpu", "auto"), "自动"))', "load")

    s = sub1(s, '            "crop_cover": self.v_ccrop.get().strip(),',
             '            "crop_cover": self.v_ccrop.get().strip(),\n'
             '            "gpu": {"自动": "auto", "只用 CPU": "cpu", "只用 GPU": "gpu"}'
             '.get(self.v_gpu.get().strip(), "auto"),', "collect")

    s = sub1(s, '        if c.get("match"):\n            argv += ["--match", c["match"]]',
             '        if c.get("match"):\n            argv += ["--match", c["match"]]\n'
             '        if c.get("gpu") and c["gpu"] != "auto":\n'
             '            argv += ["--gpu", c["gpu"]]', "argv")

    s = sub1(s, "    def _finish(self, code: int):\n        self.pb.stop()",
             "    def _finish(self, code: int):\n"
             "        try:\n"
             "            if self.mon is not None and self.mon.winfo_exists():\n"
             "                self.mon.finish(code)\n"
             "        except Exception:\n"
             "            pass\n"
             "        self.pb.stop()", "finish")

    p.write_text(s, encoding="utf-8")
    print("gui ok", p)

    # argv 里 --gpu 需要 CLI 支持（pws 已加参数，确认 flag 名）
    pp = root / "core" / "pws_params.py"
    t = pp.read_text(encoding="utf-8")
    if "'--gpu'" not in t:
        print("  注意：--gpu 的 CLI flag 由 cli_flag 自动生成，需确认 choice 参数可用")
print("done")
