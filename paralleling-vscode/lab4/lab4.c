#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <unistd.h>


#ifdef _OPENMP
#include <omp.h>
#endif

#define ITERATIONS 100
#define A 504


// OpenMP schedule settings
#ifdef _OPENMP
static omp_sched_t omp_sched = omp_sched_static;
static int omp_chunk = 0;
#endif

#ifdef _OPENMP
void parse_schedule(const char* s) {
    if (strcmp(s, "static") == 0)
        omp_sched = omp_sched_static;
    else if (strcmp(s, "dynamic") == 0)
        omp_sched = omp_sched_dynamic;
    else if (strcmp(s, "guided") == 0)
        omp_sched = omp_sched_guided;
    else {
        fprintf(stderr, "Unknown schedule: %s\n", s);
        exit(1);
    }
}
#endif

volatile int done = 0;

void progress(int total) {
    while (1) {
        int current = done;
#pragma omp critical
        printf("Progress: %.1f%%\n", 100.0 * current / total);
        if (current >= total) {
            break;
        }
        sleep(1);
    }
}

// Функция слияния двух отсортированных массивов
void merge_two_sorted(double* a, int left1, int right1,
    double* b, int left2, int right2,
    double* result, int result_start) {
    int i = left1, j = left2, k = result_start;

    while (i <= right1 && j <= right2) {
        if (a[i] <= b[j]) {
            result[k++] = a[i++];
        }
        else {
            result[k++] = b[j++];
        }
    }

    while (i <= right1) {
        result[k++] = a[i++];
    }

    while (j <= right2) {
        result[k++] = b[j++];
    }
}


// Sort
void insertion_sort(double* a, int n) {
    for (int i = 1; i < n; i++) {
        double key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = key;
    }
}

void insertion_sort_parallel_improved(double* a, int n) {
    // Если массив маленький, используем последовательную сортировку
    if (n < 1000) {
        insertion_sort(a, n);
        return;
    }

    int num_threads;
    int* starts = NULL;
    int* ends = NULL;
    double* temp = NULL;

#pragma omp parallel
    {
#pragma omp single
        {
#ifdef _OPENMP
            num_threads = omp_get_num_threads();
#endif
            starts = (int*)malloc(num_threads * sizeof(int));
            ends = (int*)malloc(num_threads * sizeof(int));

            // Вычисляем границы для каждого потока
            int chunk_size = n / num_threads;
            for (int i = 0; i < num_threads; i++) {
                starts[i] = i * chunk_size;
                ends[i] = (i == num_threads - 1) ? n : starts[i] + chunk_size;
            }
        }

        // Каждый поток сортирует свою часть
#pragma omp for schedule(static, 1)
        for (int i = 0; i < num_threads; i++) {
            int size = ends[i] - starts[i];
            insertion_sort(a + starts[i], size);
        }

        // Синхронизируем перед слиянием
#pragma omp barrier
#pragma omp single
        {
            temp = (double*)malloc(n * sizeof(double));
        }

        // Параллельное слияние пар соседних частей
        int step = 2;
        while (step / 2 < num_threads) {
#pragma omp for schedule(static)
            for (int i = 0; i < num_threads; i += step) {
                int left_part = i;
                int right_part = i + step / 2;

                // Если правой части не существует, просто копируем левую
                if (right_part >= num_threads) {
                    for (int j = starts[left_part]; j < ends[left_part]; j++) {
                        temp[j] = a[j];
                    }
                    continue;
                }

                int left_start = starts[left_part];
                int left_end = ends[left_part] - 1;
                int right_start = starts[right_part];
                int right_end = (right_part + step / 2 < num_threads) ?
                    ends[right_part] - 1 : ends[num_threads - 1] - 1;

                // Сливаем две части
                merge_two_sorted(a, left_start, left_end,
                    a, right_start, right_end,
                    temp, left_start);
            }

            // Копируем из временного массива обратно в основной
#pragma omp barrier
#pragma omp for schedule(static)
            for (int i = 0; i < n; i++) {
                a[i] = temp[i];
            }

            // Обновляем границы частей для следующего уровня слияния
#pragma omp barrier
#pragma omp single
            {
                // Обновляем границы частей
                for (int i = 0; i < num_threads; i += step) {
                    if (i + step / 2 < num_threads) {
                        ends[i] = (i + step < num_threads) ?
                            starts[i + step] : n;
                    }
                }
            }

            step *= 2;
        }

#pragma omp barrier
#pragma omp single
        {
            free(starts);
            free(ends);
            free(temp);
        }
    }
}



// Generate
unsigned int f(int it, int i) {
    return 123456u + it * 100000u + i;
}

void generate(double* M1, double* M2, int N, int it) {

#pragma omp parallel for schedule(runtime) default(none) shared(M1, N, it)
    for (int i = 0; i < N; i++) {
        unsigned int s = f(it, i);
        M1[i] = 1.0 + ((double)rand_r(&s) / RAND_MAX) * (A - 1);
    }

#pragma omp parallel for schedule(runtime) default(none) shared(M2, N, it)
    for (int i = 0; i < N / 2; i++) {
        unsigned int s = f(it, i + N);
        M2[i] = A + ((double)rand_r(&s) / RAND_MAX) * (9 * A);
    }
}

// Map
void map(double* M1, double* M2, double* copy, int N) {

#pragma omp parallel for schedule(runtime) default(none) shared(M1, N)
    for (int i = 0; i < N; i++) {
        M1[i] = exp(sqrt(M1[i]));
    }

#pragma omp parallel for schedule(runtime) default(none) shared(M2, copy, N)
        for (int i = 0; i < N / 2; i++)
            copy[i] = M2[i];

#pragma omp parallel for schedule(runtime) default(none) shared(M2, copy, N)
        for (int i = 0; i < N / 2; i++) {
            double prev = (i == 0) ? 0.0 : copy[i - 1];
            M2[i] = fabs(tan(copy[i] + prev));
        }
}

// Merge

void merge(double* M1, double* M2, int N) {

#pragma omp parallel for schedule(runtime) default(none) shared(M1, M2, N)
    for (int i = 0; i < N / 2; i++) {
        M2[i] = fmin(M1[i], M2[i]);
    }
}

// Reduce
double reduce(double* M2, int N) {
    double min = 0.0;

    for (int i = 0; i < N / 2; i++) {
        if (M2[i] != 0.0) {
            min = M2[i];
            break;
        }
    }

    double X = 0.0;
    if (min != 0.0) {

#pragma omp parallel for schedule(runtime) default(none) shared(M2, N, min) reduction(+:X)
        for (int i = 0; i < N / 2; i++) {
            int k = (int)(M2[i] / min);
            if (k % 2 == 0) {
                X += sin(M2[i]);
            }
        }
    }

    return X;
}

// Main
int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s N THREADS [schedule chunk]\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int threads = atoi(argv[2]);
    #ifdef _OPENMP
    omp_set_max_active_levels(2);
    omp_set_num_threads(threads);
    omp_set_schedule(omp_sched_static, 1);

    if (argc >= 5) {
        parse_schedule(argv[3]);
        omp_chunk = atoi(argv[4]);
        omp_set_schedule(omp_sched, omp_chunk);
    }
#endif

    double* M1 = malloc(sizeof(double) * N);
    double* M2 = malloc(sizeof(double) * (N / 2));
    double* copy = malloc(sizeof(double) * (N / 2));

    double X = 0.0;

    double start_time = 0.0, end_time = 0.0;

#ifdef _OPENMP
    start_time = omp_get_wtime();
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    start_time = tv.tv_sec + tv.tv_usec * 1e-6;
#endif

#ifdef _OPENMP
#pragma omp parallel sections num_threads(2) default(shared)
{
#pragma omp section
    {
        progress(ITERATIONS);
    }
#pragma omp section
    {
        for (int it = 0; it < ITERATIONS; it++) {
            generate(M1, M2, N, it + 1);
            map(M1, M2, copy, N);
            merge(M1, M2, N);
            insertion_sort_parallel_improved(M2, N / 2);
            X = reduce(M2, N);
#pragma omp atomic
            done++;
        }
    }       
}
#else
    for (int it = 0; it < ITERATIONS; it++) {
        generate(M1, M2, N, it + 1);
        map(M1, M2, copy, N);
        merge(M1, M2, N);
        insertion_sort_parallel_improved(M2, N / 2);
        X = reduce(M2, N);
    }
#endif

#ifdef _OPENMP
    end_time = omp_get_wtime();
#else
    gettimeofday(&tv, NULL);
    end_time = tv.tv_sec + tv.tv_usec * 1e-6;
#endif

    double ms = (end_time - start_time) * 1000.0;

    {
        const char *outfname = "auto_output_lab4.txt";
        FILE *out = fopen(outfname, "w");
        if (out) {
            fprintf(out, "%d %d %.3f %.10f\n", N, threads, ms, X);
            fclose(out);
        } else {
            fprintf(stderr, "Warning: cannot open %s for writing\n", outfname);
        }
    }

    free(M1);
    free(M2);
    return 0;
}