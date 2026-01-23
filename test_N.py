import subprocess
import os
import sys
import time
import threading
import psutil
import matplotlib.pyplot as plt
from collections import defaultdict

# НАСТРОЙКИ

N_VALUES = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]
THREADS = [1, 2, 4, 8, 12]

LABS = {
    "LR1": "./lab1",
    "LR3": "./lab3",
}

RESULTS_DIR = "plots_n"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ

def run_cmd(cmd):
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
    parts = out.split()

    if len(parts) == 3:          # LR1
        _, time_ms, _ = parts
        threads = 1
    elif len(parts) == 4:        # LR3
        _, threads, time_ms, _ = parts
        threads = int(threads)
    else:
        raise RuntimeError(f"Unexpected output format:\n{out}")

    return threads, float(time_ms)


def save_plot(x, ys, labels, title, ylabel, filename):
    plt.figure()
    for y, label in zip(ys, labels):
        plt.plot(x, y, marker='o', label=label)
    plt.xlabel("N")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()

# СБОР ВРЕМЕНИ ВЫПОЛНЕНИЯ

times = defaultdict(lambda: defaultdict(dict))
# times[lab][threads][N] = time_ms

print("== Running experiments for different N ==")

for N in N_VALUES:
    print(f"\n--- N = {N} ---")

    # LR1
    threads, time_ms = run_cmd([LABS["LR1"], str(N)])
    times["LR1"][1][N] = time_ms
    print(f"LR1: time={time_ms:.2f} ms")

    # LR3
    for t in THREADS:
        threads, time_ms = run_cmd([LABS["LR3"], str(N), str(t)])
        times["LR3"][t][N] = time_ms
        print(f"LR3: threads={t}, time={time_ms:.2f} ms")

# ВРЕМЯ ВЫПОЛНЕНИЯ vs N

ys = []
labels = []

ys.append([times["LR1"][1][N] for N in N_VALUES])
labels.append("LR1 (1 thread)")

for t in THREADS:
    ys.append([times["LR3"][t][N] for N in N_VALUES])
    labels.append(f"LR3 ({t} threads)")

save_plot(
    N_VALUES,
    ys,
    labels,
    "Execution time vs N",
    "Time, ms",
    "time_vs_N.png"
)

# УСКОРЕНИЕ vs N (LR3)

ys = []
labels = []

for t in THREADS:
    if t == 1:
        continue
    sp = [
        times["LR3"][1][N] / times["LR3"][t][N]
        for N in N_VALUES
    ]
    ys.append(sp)
    labels.append(f"{t} threads")

save_plot(
    N_VALUES,
    ys,
    labels,
    "Speedup vs N (LR3)",
    "Speedup",
    "speedup_vs_N.png"
)

# ЭФФЕКТИВНОСТЬ vs N (LR3)

ys = []
labels = []

for t in THREADS:
    if t == 1:
        continue
    eff = [
        (times["LR3"][1][N] / times["LR3"][t][N]) / t
        for N in N_VALUES
    ]
    ys.append(eff)
    labels.append(f"{t} threads")

save_plot(
    N_VALUES,
    ys,
    labels,
    "Efficiency vs N (LR3)",
    "Efficiency",
    "efficiency_vs_N.png"
)

# ЗАГРУЗКА ЯДЕР CPU

print("\n== CPU load profiling ==")

BEST_N = max(N_VALUES)
BEST_THREADS = max(THREADS)

cmd = [
    LABS["LR3"],
    str(BEST_N),
    str(BEST_THREADS)
]

cpu_data = []
timestamps = []

def monitor_cpu(proc, interval=0.1):
    start = time.time()
    while proc.poll() is None:
        cpu = psutil.cpu_percent(interval=interval, percpu=True)
        cpu_data.append(cpu)
        timestamps.append(time.time() - start)

proc = subprocess.Popen(cmd)
monitor_thread = threading.Thread(target=monitor_cpu, args=(proc,))
monitor_thread.start()

proc.wait()
monitor_thread.join()

plt.figure(figsize=(10, 6))
num_cores = len(cpu_data[0])

for core in range(num_cores):
    plt.plot(
        timestamps,
        [sample[core] for sample in cpu_data],
        label=f"CPU {core}"
    )

plt.xlabel("Time, seconds")
plt.ylabel("CPU usage, %")
plt.title(
    "CPU core utilization over time\n"
    f"N={BEST_N}, threads={BEST_THREADS}"
)
plt.grid()
plt.legend(fontsize="small")

plt.savefig(os.path.join(RESULTS_DIR, "cpu_load_over_time.png"))
plt.close()

print("CPU load graph saved: cpu_load_over_time.png")
print("\nAll experiments finished.")
print(f"Plots saved to ./{RESULTS_DIR}/")
