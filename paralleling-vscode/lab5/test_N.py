import subprocess
import os
import sys
import time
import threading
import psutil
import matplotlib.pyplot as plt
from collections import defaultdict

# НАСТРОЙКИ

NS = [100, 500, 1000, 5000, 10000, 50000, 100000]
THREADS = [1, 2, 4, 6, 8]

LABS = {
    "LR4": "./lab4",
    "LR5": "./lab5",
}

RESULTS_DIR = "plots"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ

def run_cmd(cmd):

    exe_base = os.path.splitext(os.path.basename(cmd[0]))[0]
    auto_fname = f"auto_output_{exe_base}.txt"
    
    try:
        if os.path.exists(auto_fname):
            os.remove(auto_fname)
    except OSError:
        pass

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    content = None
    if os.path.exists(auto_fname):
        try:
            with open(auto_fname, "r") as f:
                content = f.read().strip()
        except Exception:
            content = proc.stdout.strip()
    else:
        content = proc.stdout.strip()

    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    parts = None
    for ln in reversed(lines):
        toks = ln.split()
        if len(toks) == 4:
            parts = toks
            break

    if parts is None:
        raise RuntimeError(f"Unexpected output format. Raw output:\n{content}")

    _, threads, time_ms, _ = parts
    return int(threads), float(time_ms)


def save_plot(x, ys, labels, title, ylabel, filename):
    plt.figure()
    for y, label in zip(ys, labels):
        plt.plot(x, y, marker='o', label=label)
    plt.xlabel("Threads" if isinstance(x[0], int) else "N")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()


# ЗАМЕР ВРЕМЕНИ

times = defaultdict(lambda: defaultdict(dict))
# times[lab][N][threads] = time

print("== Running benchmarks ==")

for lab, exe in LABS.items():
    if not os.path.exists(exe):
        print(f"Executable {exe} not found")
        sys.exit(1)

    for N in NS:
        for t in THREADS:
            cmd = [exe, str(N), str(t)]
            threads, time_ms = run_cmd(cmd)
            times[lab][N][threads] = time_ms
            print(f"{lab}: N={N}, threads={threads}, time={time_ms:.2f} ms")

# ГРАФИКИ ВРЕМЕНИ ОТ N

for t in THREADS:
    ys = []
    labels = []

    for lab in LABS:
        ys.append([times[lab][N][t] for N in NS])
        labels.append(f"{lab}, threads={t}")

    save_plot(
        NS,
        ys,
        labels,
        f"Execution time vs N (threads={t})",
        "Time, ms",
        f"time_vs_N_threads_{t}.png"
    )

# УСКОРЕНИЕ

for N in NS:
    ys = []
    labels = []

    for lab in LABS:
        t1 = times[lab][N][1]
        sp = [t1 / times[lab][N][p] for p in THREADS]
        ys.append(sp)
        labels.append(lab)

    save_plot(
        THREADS,
        ys,
        labels,
        f"Speedup vs Threads (N={N})",
        "Speedup",
        f"speedup_N_{N}.png"
    )

# ЭФФЕКТИВНОСТЬ

for N in NS:
    ys = []
    labels = []

    for lab in LABS:
        t1 = times[lab][N][1]
        eff = [(t1 / times[lab][N][p]) / p for p in THREADS]
        ys.append(eff)
        labels.append(lab)

    save_plot(
        THREADS,
        ys,
        labels,
        f"Efficiency vs Threads (N={N})",
        "Efficiency",
        f"efficiency_N_{N}.png"
    )

# ПРЯМОЕ СРАВНЕНИЕ LAB4 vs LAB5

for N in NS:
    ratio = [
        times["LR5"][N][p] / times["LR4"][N][p]
        for p in THREADS
    ]

    save_plot(
        THREADS,
        [ratio],
        ["LR5 / LR4"],
        f"Relative performance (N={N})",
        "Time ratio",
        f"relative_perf_N_{N}.png"
    )


# ЗАГРУЗКА CPU ПО ВРЕМЕНИ

BEST_N = max(NS)
BEST_THREADS = max(THREADS)

def profile_cpu(lab):
    cmd = [
        LABS[lab],
        str(BEST_N),
        str(BEST_THREADS)
    ]

    cpu_data = []
    timestamps = []

    def monitor(proc, interval=0.1):
        start = time.time()
        while proc.poll() is None:
            cpu = psutil.cpu_percent(interval=interval, percpu=True)
            cpu_data.append(cpu)
            timestamps.append(time.time() - start)

    proc = subprocess.Popen(cmd)
    t = threading.Thread(target=monitor, args=(proc,))
    t.start()
    proc.wait()
    t.join()

    plt.figure(figsize=(10, 6))
    for core in range(len(cpu_data[0])):
        plt.plot(
            timestamps,
            [sample[core] for sample in cpu_data],
            label=f"CPU {core}"
        )

    plt.xlabel("Time, s")
    plt.ylabel("CPU usage, %")
    plt.title(f"CPU load over time ({lab})")
    plt.grid()
    plt.legend(fontsize="small")
    plt.savefig(os.path.join(RESULTS_DIR, f"cpu_load_{lab}.png"))
    plt.close()


print("\n== CPU profiling ==")
profile_cpu("LR4")
profile_cpu("LR5")

print("\nAll experiments finished.")
print(f"Plots saved to ./{RESULTS_DIR}/")
