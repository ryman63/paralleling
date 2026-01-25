import subprocess
import os
import sys
import time
import threading
import psutil
import matplotlib.pyplot as plt
from collections import defaultdict

# НАСТРОЙКИ
INPUT_FILE = "results.txt"  # Файл с результатами запусков
LABS = {
    "LR4": "./lab4",
    "LR5": "./lab5",
}
RESULTS_DIR = "plots"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ

def parse_results_file(filename):
    """
    Парсит файл с результатами.
    Ожидаемый формат строки:
    LR4: N=1000, threads=4, time=12.34 ms
    или
    lab_name N threads time_ms
    """
    times = defaultdict(lambda: defaultdict(dict))
    # times[lab][N][threads] = time_ms
    
    if not os.path.exists(filename):
        print(f"Файл {filename} не найден!")
        print("Запустите сначала сбор данных командой:")
        print("python collect_data.py")
        sys.exit(1)
    
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Пробуем разные форматы
            if ':' in line:
                # Формат: LR4: N=1000, threads=4, time=12.34 ms
                try:
                    lab_part, data_part = line.split(':', 1)
                    lab = lab_part.strip()
                    data = {}
                    for item in data_part.split(','):
                        if '=' in item:
                            key, value = item.strip().split('=')
                            data[key.strip()] = value.strip()
                    
                    N = int(data.get('N', 0))
                    threads = int(data.get('threads', 0))
                    time_str = data.get('time', '0')
                    time_ms = float(time_str.split()[0])  # Берем только число
                    
                    times[lab][N][threads] = time_ms
                    
                except Exception as e:
                    print(f"Ошибка в строке {line_num}: {line}")
                    print(f"Ошибка: {e}")
                    continue
                    
            elif line.count(' ') >= 3:
                # Формат: LR4 1000 4 12.34
                try:
                    parts = line.split()
                    lab = parts[0]
                    N = int(parts[1])
                    threads = int(parts[2])
                    time_ms = float(parts[3])
                    
                    times[lab][N][threads] = time_ms
                    
                except Exception as e:
                    print(f"Ошибка в строке {line_num}: {line}")
                    print(f"Ошибка: {e}")
                    continue
    
    return times

def collect_data():
    """
    Собирает данные запусков в файл (если нужно)
    """
    NS = [100, 500, 1000, 5000, 10000, 50000, 100000]
    THREADS = [1, 2, 4, 8, 12]
    
    print("== Сбор данных запусков ==")
    print(f"Результаты будут сохранены в {INPUT_FILE}")
    
    with open(INPUT_FILE, 'w') as f:
        f.write("# Формат: lab_name N threads time_ms\n")
        f.write("# или lab_name: N=value, threads=value, time=value ms\n\n")
        
        for lab, exe in LABS.items():
            if not os.path.exists(exe):
                print(f"Исполняемый файл {exe} не найден")
                sys.exit(1)
            
            for N in NS:
                for t in THREADS:
                    cmd = [exe, str(N), str(t)]
                    
                    exe_base = os.path.splitext(os.path.basename(cmd[0]))[0]
                    auto_fname = f"auto_output_{exe_base}.txt"
                    
                    try:
                        if os.path.exists(auto_fname):
                            os.remove(auto_fname)
                    except OSError:
                        pass

                    print(f"Запуск: {lab}, N={N}, threads={t}")
                    
                    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

                    content = None
                    if os.path.exists(auto_fname):
                        try:
                            with open(auto_fname, "r") as auto_file:
                                content = auto_file.read().strip()
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
                        print(f"  Ошибка: неожиданный вывод программы")
                        print(f"  Вывод: {content}")
                        continue

                    _, threads_actual, time_ms, _ = parts
                    threads_actual = int(threads_actual)
                    time_ms = float(time_ms)
                    
                    # Сохраняем в оба формата для совместимости
                    f.write(f"{lab}: N={N}, threads={threads_actual}, time={time_ms:.2f} ms\n")
                    f.write(f"{lab} {N} {threads_actual} {time_ms:.2f}\n\n")
                    
                    print(f"  Результат: {time_ms:.2f} ms")

def save_plot(x, ys, labels, title, ylabel, filename):
    """Сохраняет график в файл"""
    plt.figure(figsize=(10, 6))
    for y, label in zip(ys, labels):
        plt.plot(x, y, marker='o', label=label, linewidth=2)
    plt.xlabel("Threads" if isinstance(x[0], int) else "N", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Сохраняем с высоким качеством
    filepath = os.path.join(RESULTS_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  График сохранен: {filename}")

def analyze_data(times):
    """
    Анализирует данные и строит графики
    """
    print("\n== Анализ данных и построение графиков ==")
    
    # Определяем доступные N и threads из данных
    labs = list(times.keys())
    if not labs:
        print("Нет данных для анализа!")
        return
    
    # Получаем все уникальные N и threads
    all_N = set()
    all_threads = set()
    for lab in labs:
        for N in times[lab]:
            all_N.add(N)
            for threads in times[lab][N]:
                all_threads.add(threads)
    
    NS = sorted(all_N)
    THREADS = sorted(all_threads)
    
    print(f"Найдено N: {NS}")
    print(f"Найдено threads: {THREADS}")
    
    # 1. Графики времени от N
    print("\n1. Графики времени выполнения от N:")
    for t in THREADS:
        ys = []
        labels = []
        
        for lab in labs:
            if any(t in times[lab][N] for N in NS):
                y = []
                for N in NS:
                    if t in times[lab][N]:
                        y.append(times[lab][N][t])
                    else:
                        y.append(None)
                ys.append(y)
                labels.append(f"{lab}, threads={t}")
        
        if ys and labels:
            # Фильтруем только те N, для которых есть данные
            valid_indices = [i for i, N in enumerate(NS) if all(y[i] is not None for y in ys)]
            if valid_indices:
                valid_NS = [NS[i] for i in valid_indices]
                valid_ys = [[y[i] for i in valid_indices] for y in ys]
                
                save_plot(
                    valid_NS,
                    valid_ys,
                    labels,
                    f"Время выполнения от N (threads={t})",
                    "Время, мс",
                    f"time_vs_N_threads_{t}.png"
                )
    
    # 2. Ускорение
    print("\n2. Графики ускорения (Speedup):")
    for N in NS:
        ys = []
        labels = []
        
        for lab in labs:
            if 1 in times[lab][N] and any(p in times[lab][N] for p in THREADS):
                t1 = times[lab][N][1]
                sp = []
                for p in THREADS:
                    if p in times[lab][N]:
                        sp.append(t1 / times[lab][N][p])
                    else:
                        sp.append(None)
                ys.append(sp)
                labels.append(lab)
        
        if ys and labels:
            # Фильтруем только те threads, для которых есть данные
            valid_indices = [i for i, t in enumerate(THREADS) if all(y[i] is not None for y in ys)]
            if valid_indices:
                valid_THREADS = [THREADS[i] for i in valid_indices]
                valid_ys = [[y[i] for i in valid_indices] for y in ys]
                
                save_plot(
                    valid_THREADS,
                    valid_ys,
                    labels,
                    f"Ускорение от количества потоков (N={N})",
                    "Ускорение",
                    f"speedup_N_{N}.png"
                )
    
    # 3. Эффективность
    print("\n3. Графики эффективности (Efficiency):")
    for N in NS:
        ys = []
        labels = []
        
        for lab in labs:
            if 1 in times[lab][N] and any(p in times[lab][N] for p in THREADS):
                t1 = times[lab][N][1]
                eff = []
                for p in THREADS:
                    if p in times[lab][N]:
                        eff.append((t1 / times[lab][N][p]) / p)
                    else:
                        eff.append(None)
                ys.append(eff)
                labels.append(lab)
        
        if ys and labels:
            valid_indices = [i for i, t in enumerate(THREADS) if all(y[i] is not None for y in ys)]
            if valid_indices:
                valid_THREADS = [THREADS[i] for i in valid_indices]
                valid_ys = [[y[i] for i in valid_indices] for y in ys]
                
                save_plot(
                    valid_THREADS,
                    valid_ys,
                    labels,
                    f"Эффективность от количества потоков (N={N})",
                    "Эффективность",
                    f"efficiency_N_{N}.png"
                )
    
    # 4. Сравнение LR4 и LR5
    print("\n4. Сравнение LR4 и LR5:")
    if "LR4" in labs and "LR5" in labs:
        for N in NS:
            if all(p in times["LR4"][N] and p in times["LR5"][N] for p in THREADS):
                ratio = [
                    times["LR5"][N][p] / times["LR4"][N][p]
                    for p in THREADS
                ]
                
                save_plot(
                    THREADS,
                    [ratio],
                    ["LR5 / LR4"],
                    f"Относительная производительность (N={N})",
                    "Отношение времени (LR5/LR4)",
                    f"relative_perf_N_{N}.png"
                )

def profile_cpu():
    """
    Профилирование загрузки CPU (если нужно)
    """
    print("\n== Профилирование загрузки CPU ==")
    
    # Находим максимальные N и threads из данных
    times = parse_results_file(INPUT_FILE)
    
    if not times:
        print("Нет данных для профилирования!")
        return
    
    # Находим максимальные значения
    max_N = 0
    max_threads = 0
    for lab in times:
        for N in times[lab]:
            if N > max_N:
                max_N = N
            for threads in times[lab][N]:
                if threads > max_threads:
                    max_threads = threads
    
    if max_N == 0 or max_threads == 0:
        print("Не удалось определить параметры для профилирования!")
        return
    
    BEST_N = max_N
    BEST_THREADS = max_threads
    
    print(f"Используем для профилирования: N={BEST_N}, threads={BEST_THREADS}")
    
    for lab, exe in LABS.items():
        if not os.path.exists(exe):
            print(f"Исполняемый файл {exe} не найден")
            continue
        
        cmd = [exe, str(BEST_N), str(BEST_THREADS)]
        
        cpu_data = []
        timestamps = []
        
        def monitor(proc, interval=0.1):
            start = time.time()
            while proc.poll() is None:
                cpu = psutil.cpu_percent(interval=interval, percpu=True)
                cpu_data.append(cpu)
                timestamps.append(time.time() - start)
        
        print(f"\nПрофилирование {lab}...")
        proc = subprocess.Popen(cmd)
        t = threading.Thread(target=monitor, args=(proc,))
        t.start()
        proc.wait()
        t.join()
        
        if not cpu_data:
            print(f"  Нет данных о загрузке CPU для {lab}")
            continue
        
        plt.figure(figsize=(12, 8))
        for core in range(len(cpu_data[0])):
            plt.plot(
                timestamps,
                [sample[core] for sample in cpu_data],
                label=f"Ядро {core}",
                linewidth=1.5,
                alpha=0.7
            )
        
        plt.xlabel("Время, с", fontsize=12)
        plt.ylabel("Загрузка CPU, %", fontsize=12)
        plt.title(f"Загрузка CPU по времени ({lab}, N={BEST_N}, threads={BEST_THREADS})", 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize="small", ncol=2)
        plt.tight_layout()
        
        filename = f"cpu_load_{lab}.png"
        filepath = os.path.join(RESULTS_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  График сохранен: {filename}")

def main():
    """Основная функция"""
    print("=" * 60)
    print("АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ ПАРАЛЛЕЛЬНЫХ ПРОГРАММ")
    print("=" * 60)
    
    # Проверяем, нужно ли собирать данные
    if not os.path.exists(INPUT_FILE):
        print(f"Файл {INPUT_FILE} не найден.")
        collect = input("Собрать данные запусков? (y/n): ").lower().strip()
        if collect == 'y':
            collect_data()
        else:
            print("Для анализа нужен файл с данными!")
            sys.exit(1)
    else:
        print(f"Найден файл с данными: {INPUT_FILE}")
        use_existing = input("Использовать существующие данные? (y/n): ").lower().strip()
        if use_existing != 'y':
            collect_data()
    
    # Читаем и анализируем данные
    times = parse_results_file(INPUT_FILE)
    
    if not times:
        print("Нет данных для анализа!")
        sys.exit(1)
    
    # Анализируем данные
    analyze_data(times)
    
    # Профилирование CPU (опционально)
    profile_cpu_choice = input("\nВыполнить профилирование загрузки CPU? (y/n): ").lower().strip()
    if profile_cpu_choice == 'y':
        profile_cpu()
    
    print("\n" + "=" * 60)
    print("ВСЕ ОПЕРАЦИИ ЗАВЕРШЕНЫ!")
    print(f"Графики сохранены в папке: ./{RESULTS_DIR}/")
    print("=" * 60)

if __name__ == "__main__":
    main()