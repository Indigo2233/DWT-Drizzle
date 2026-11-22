# -*- coding: utf-8 -*-
"""打发行包：A 四包 + C 一包。

只做一件事：把目录树按排除规则压成 zip（UTF-8 文件名、保留 mac 可执行位），
然后把包内根目录名统一成约定的名字。

用法：
    python build_zips.py            # 打全部
    python build_zips.py A          # 只打 A
    python build_zips.py C          # 只打 C
"""
import os
import stat
import sys
import zipfile
from pathlib import Path

DIST = Path(r"G:/DWT/dist")
SKIP_DIRS = {"_DWT_frames", "__pycache__", ".git"}
SKIP_FILES = {".DS_Store"}
SKIP_SUFFIX = (".broken", ".pyc", ".dat")
EXEC_SUFFIX = (".command", ".so", ".dylib", ".sh")

JOBS = {
    "A": [
        (DIST / "DWT-Drizzle-2.7", DIST / "DWT-Drizzle-2.7.zip", "DWT-Drizzle-2.7", False),
        (DIST / "DWT-Drizzle-2.7-macos", DIST / "DWT-Drizzle-2.7-macos.zip",
         "DWT-Drizzle-2.7-macos", True),
        (DIST / "DWT-Drizzle-2.7-macos-arm64", DIST / "DWT-Drizzle-2.7-macos-arm64.zip",
         "DWT-Drizzle-2.7-macos-arm64", True),
        (DIST / "DWT-Drizzle-2.7-macos-intel", DIST / "DWT-Drizzle-2.7-macos-intel.zip",
         "DWT-Drizzle-2.7-macos-intel", True),
    ],
    "C": [
        (DIST / "DWT-Drizzle-C", DIST / "DWT-Drizzle-C-2.7.48-win64.zip",
         "DWT-Drizzle-C", False),
    ],
}


def preflight(tree: Path) -> bool:
    """打包前跑随包自测：磁盘预检 + 彩机/黑白隔离。不通过就别打。"""
    import subprocess
    exe = tree / "py" / "python.exe"
    if not exe.exists():
        exe = Path(sys.executable)
    ok = True
    for name, tag in (("selftest_disk_guard.py", "DISK-GUARD-OK"),
                      ("selftest_mono_isolation.py", "MONO-ISOLATION-OK")):
        f = tree / name
        if not f.exists():
            print("  [跳过] 包里没有", name)
            continue
        r = subprocess.run([str(exe), str(f)], cwd=str(tree),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = r.stdout.decode("gbk", "replace")
        good = tag in out
        print("  [自测] %-32s %s" % (name, "PASS" if good else "FAIL"))
        if not good:
            print(out[-800:])
            ok = False
    return ok


def build(tree: Path, out: Path, root: str, mac: bool, keep_gpudeps: bool = True):
    n = 0
    total = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(tree.rglob("*")):
            rel = p.relative_to(tree)
            if any(part in SKIP_DIRS for part in rel.parts):
                continue
            if p.is_dir() or p.name in SKIP_FILES or p.name.endswith(SKIP_SUFFIX):
                continue
            arc = root + "/" + "/".join(rel.parts)
            zi = zipfile.ZipInfo.from_file(p, arc)
            zi.compress_type = zipfile.ZIP_DEFLATED
            if mac and (p.suffix in EXEC_SUFFIX or p.name.startswith("启动")):
                zi.external_attr = (stat.S_IFREG | 0o755) << 16
            with open(p, "rb") as fh:
                zf.writestr(zi, fh.read())
            n += 1
            total += p.stat().st_size
    print("  %-46s %6.1f MB zip / %6.1f MB raw | %d 个文件"
          % (out.name, out.stat().st_size / 1e6, total / 1e6, n))


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "AC"
    for key in which:
        for tree, out, root, mac in JOBS[key]:
            if not tree.exists():
                print("  [跳过] 没有目录", tree)
                continue
            if not preflight(tree):
                print("  [中止] 自测未通过，不打包", tree)
                return 2
            if out.exists():
                out.unlink()
            build(tree, out, root, mac)


if __name__ == "__main__":
    main()
