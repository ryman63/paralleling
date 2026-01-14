import subprocess
import os
import sys
import time
import threading
import psutil
import matplotlib.pyplot as plt
from collections import defaultdict

# =========================================================
# НАСТРОЙКИ
# =========================================================

N = 100_000            # N1 из задания
THREADS = [1, 2, 4, 6, 8]

LABS = {
    "LR1": "./lab1",
    "LR3": "./lab3",
}

SCHEDULES = ["static", "dynamic", "guided"]

RESULTS_DIR = "plots"
os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================================================

def run_cmd(cmd):
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
    parts = out.split()

    if len(parts) == 3:        # LR1
        _, time_ms, _ = parts
        threads = 1
    elif len(parts) == 4:      # LR3
        _, threads, time_ms, _ = parts
        threads = int(threads)
    else:
        raise RuntimeError("Unexpected output format")

    return threads, float(time_ms)


def save_plot(x, ys, labels, title, ylabel, filename):
    plt.figure()
    for y, label in zip(ys, labels):
        plt.plot(x, y, marker='o', label=label)
    plt.xlabel("Threads")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()


# =========================================================
# 1. СРАВНЕНИЕ ЛР1 / ЛР3
# =========================================================

times = defaultdict(dict)

print("== Running LR1 / LR3 ==")

for lab, exe in LABS.items():
    if not os.path.exists(exe):
        print(f"Executable {exe} not found")
        sys.exit(1)

    for t in THREADS:
        if lab == "LR1":
            cmd = [exe, str(N)]
        else:
            cmd = [exe, str(N), str(t)]

        threads, time_ms = run_cmd(cmd)
        times[lab][threads] = time_ms
        print(f"{lab}: threads={threads}, time={time_ms:.2f} ms")

speedups = {
    lab: {p: times[lab][1] / times[lab][p] for p in times[lab]}
    for lab in times
}

save_plot(
    THREADS,
    [[speedups[lab].get(t, 1) for t in THREADS] for lab in LABS],
    list(LABS.keys()),
    "Parallel speedup comparison (LR1 vs LR3)",
    "Speedup",
    "speedup_lr1_lr3.png"
)

# =========================================================
# 2. SCHEDULE + CHUNK
# =========================================================

schedule_results = defaultdict(lambda: defaultdict(dict))

print("\n== Running schedule experiments ==")

for sched in SCHEDULES:
    for t in THREADS:
        chunk_variants = {
            "1": 1,
            "<P": max(1, t // 2),
            "=P": t,
            ">P": t * 2
        }

        for cname, chunk in chunk_variants.items():
            cmd = [
                LABS["LR3"],
                str(N),
                str(t),
                sched,
                str(chunk)
            ]
            _, time_ms = run_cmd(cmd)
            schedule_results[sched][cname][t] = time_ms
            print(f"{sched}, chunk={chunk}, threads={t}, time={time_ms:.2f}")

# =========================================================
# 3. ГРАФИКИ УСКОРЕНИЯ
# =========================================================

for cname in ["1", "<P", "=P", ">P"]:
    ys = []
    labels = []

    for sched in SCHEDULES:
        t1 = schedule_results[sched][cname][1]
        sp = [t1 / schedule_results[sched][cname][t] for t in THREADS]
        ys.append(sp)
        labels.append(sched)

    save_plot(
        THREADS,
        ys,
        labels,
        f"Speedup for different schedules (chunk {cname})",
        "Speedup",
        f"speedup_schedule_chunk_{cname}.png"
    )

# =========================================================
# 4. ЭФФЕКТИВНОСТЬ
# =========================================================

for sched in SCHEDULES:
    ys = []
    labels = []

    for cname in ["1", "<P", "=P", ">P"]:
        t1 = schedule_results[sched][cname][1]
        sp = [t1 / schedule_results[sched][cname][t] for t in THREADS]
        eff = [sp[i] / THREADS[i] for i in range(len(THREADS))]
        ys.append(eff)
        labels.append(f"chunk {cname}")

    save_plot(
        THREADS,
        ys,
        labels,
        f"Efficiency for schedule {sched}",
        "Efficiency",
        f"efficiency_{sched}.png"
    )

# =========================================================
# 5. ГРАФИК ЗАГРУЗКИ ЯДЕР ОТ ВРЕМЕНИ (ОБЯЗАТЕЛЬНО)
# =========================================================

print("\n== CPU load profiling ==")

BEST_THREADS = max(THREADS)
BEST_SCHEDULE = "static"
BEST_CHUNK = 1

cmd = [
    LABS["LR3"],
    str(N),
    str(BEST_THREADS),
    BEST_SCHEDULE,
    str(BEST_CHUNK)
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
    f"CPU core utilization over time\n"
    f"N={N}, threads={BEST_THREADS}, "
    f"{BEST_SCHEDULE}, chunk={BEST_CHUNK}"
)
plt.grid()
plt.legend(fontsize="small")

plt.savefig(os.path.join(RESULTS_DIR, "cpu_load_over_time.png"))
plt.close()

print("CPU load graph saved: cpu_load_over_time.png")
print("\nAll experiments finished.")
print(f"Plots saved to ./{RESULTS_DIR}/")
