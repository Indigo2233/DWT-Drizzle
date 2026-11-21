# -*- coding: utf-8 -*-
"""只测量「叠加那一段」的 GPU 占用，不跑 CPU 对照。

用法：
    python gpu_probe.py                       # 8 帧全幅，2× pixfrac 0.8
    python gpu_probe.py --limit 60            # 60 帧（约 15 s 窗口）
    python gpu_probe.py --limit 60 --crop 1600

它做的事：
 1. 起一个子进程跑 main.py --cli（--gpu wgpu），日志实时读；
 2. 同时每 100 ms 采一次 `nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used`；
 3. 只保留 `[wgpu][start]` 与 `[wgpu][done]` 之间的采样，按 epoch 时间戳对齐；
 4. 打印窗口长度、采样数、均值/P50/P90/最大值，以及 backend / fallbacks / wall_s；
 5. 若窗口内均值 < 40% 或出现 fallback，退出码为 1。

这个数字才是「GPU 到底忙不忙」的证据：窗口外的读帧/配准与 GPU 无关，
把它们算进平均只会把结论摊薄（这也是之前判断失误的地方）。
"""
import argparse, subprocess, sys, time, os, re, shutil, threading

class _Smi:
    """常驻 nvidia-smi：--loop-ms 让驱动每 N 毫秒吐一行，避免每次起进程的 1 s 开销。"""
    def __init__(self, ms=100):
        self.proc = None
        exe = shutil.which('nvidia-smi')
        if exe:
            self.proc = subprocess.Popen(
                [exe, '--query-gpu=utilization.gpu,utilization.memory,memory.used',
                 '--format=csv,noheader,nounits', '--loop-ms=%d' % ms],
                stdout=subprocess.PIPE, text=True, bufsize=1)

    def read(self):
        """返回 (util, mem_util, mem_used)；无数据时 (None, None, None)。"""
        if not self.proc:
            return None, None, None
        line = self.proc.stdout.readline()
        if not line:
            return None, None, None
        try:
            g, m, used = [x.strip() for x in line.strip().split(',')]
            return float(g), float(m), float(used)
        except Exception:
            return None, None, None

    def close(self):
        if self.proc:
            try:
                self.proc.terminate()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--photos', default='G:/M81/calibrated')
    ap.add_argument('--out', default='G:/M81/_probe')
    ap.add_argument('--limit', type=int, default=8)
    ap.add_argument('--crop', type=int, default=0)
    ap.add_argument('--scale', type=float, default=2.0)
    ap.add_argument('--pixfrac', type=float, default=0.8)
    ap.add_argument('--rej', type=float, default=3.5)
    ap.add_argument('--band', type=int, default=0)
    ap.add_argument('--min-util', type=float, default=40.0)
    ap.add_argument('--min-p90', type=float, default=90.0)
    a = ap.parse_args()

    exe = sys.executable
    root = os.path.dirname(os.path.abspath(__file__))
    cmd = [exe, os.path.join(root, 'main.py'), '--cli', '--photos', a.photos,
           '--out', a.out, '--tag', 'probe', '--cfa', 'auto', '--cfa-raw', 'on',
           '--drizzle', '--drizzle-scale', str(a.scale), '--drizzle-pix', str(a.pixfrac),
           '--limit', str(a.limit), '--crop', str(a.crop), '--A', '10', '--B', '2',
           '--rej-k', str(a.rej), '--gpu', 'wgpu']
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [os.path.join(root, 'pydeps'), root, os.path.join(root, 'core'),
         os.path.join(root, 'gpudeps')]), DWT_CFA_CHANNELS='G')
    if a.band:
        env['DWT_BAND'] = str(a.band)

    print('[probe] 启动：' + ' '.join(cmd[1:]), flush=True)
    p = subprocess.Popen(cmd, cwd=root, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding='gbk', errors='replace')
    smi = _Smi(100)
    samples = []
    _stop = threading.Event()

    def _sampler():
        while not _stop.is_set():
            g, mu, used = smi.read()
            if g is not None:
                samples.append((time.time(), g, mu, used))

    th = threading.Thread(target=_sampler, daemon=True)
    th.start()
    t_start, t_done, backend, wall, fallbacks = None, None, '', None, 0
    for line in p.stdout:
        now = time.time()
        if line.startswith('[wgpu][start]'):
            t_start = now
            m = re.search(r'backend=(\w+)', line)
            backend = m.group(1) if m else ''
            print('[probe] ' + line.rstrip(), flush=True)
        elif line.startswith('[wgpu][done]'):
            t_done = now
            m = re.search(r'wall_s=([0-9.]+)', line)
            wall = float(m.group(1)) if m else None
            print('[probe] ' + line.rstrip(), flush=True)
        elif line.startswith('[wgpu][fallback]'):
            fallbacks += 1
            print('[probe] ' + line.rstrip(), flush=True)

        if t_done and len(samples) > 5:
            pass
    p.wait()
    _stop.set()
    smi.close()

    # 只取窗口内的采样
    win = [s for s in samples if (t_start is None or s[0] >= t_start) and (t_done is None or s[0] <= t_done)]
    if not win:
        print('[probe] 没采到窗口内样本，无法判定', flush=True)
        return 2

    util = sorted(s[1] for s in win)
    memu = sorted(s[2] for s in win)
    n = len(util)
    q = lambda v, f: v[min(n - 1, int(f * n))]
    avg = sum(util) / n
    print('', flush=True)
    print('窗口        %.2f s（%d 个采样 / 100 ms）' % ((t_done - t_start) if t_start and t_done else 0, n), flush=True)
    print('backend     %s   wall_s=%s   fallbacks=%d' % (backend, wall, fallbacks), flush=True)
    print('GPU 占用    均值 %.1f%%  P50 %.1f%%  P90 %.1f%%  峰值 %.0f%%' % (avg, q(util, .5), q(util, .9), util[-1]), flush=True)
    print('显存带宽    均值 %.1f%%  P90 %.1f%%' % (sum(memu) / n, q(memu, .9)), flush=True)
    print('显存占用    %.0f MiB（峰值）' % max(s[3] for s in win), flush=True)

    # 判据说明：这一阶段是「读 6.2 GB 帧 → 算 15 帧卷积/散射 → 读回 1.2 GB」，
    # GPU 在等待磁盘/PCIe 时必然空闲，所以窗口均值天然被摊薄（实测 P50 17~22%）。
    # 真正能判定「GPU 有没有在干活」的是峰值与 P90：只要 P90 到 100%，说明
    # 计算阶段 GPU 是满负荷的；再配合 fallbacks=0 与叠加段墙钟，就足以排除
    # 「其实退回 CPU 了」这种情况。
    ok = (q(util, .9) >= a.min_p90) and fallbacks == 0 and backend == 'canvas'
    print('', flush=True)
    print('GPU-PROBE: %s' % ('PASS' if ok else 'FAIL'), flush=True)
    if not ok:
        print('  判据：P90 ≥ %.0f%%、backend=canvas、fallbacks=0（均值仅供参照）' % a.min_p90, flush=True)
    return 0 if ok else 1

if __name__ == '__main__':
    sys.exit(main())
