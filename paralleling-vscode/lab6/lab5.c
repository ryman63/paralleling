#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define ITERATIONS 100
#define A 504

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

void merge_two_sorted(double* a, int l1, int r1,
    double* b, int l2, int r2,
    double* res, int start) {
    int i = l1, j = l2, k = start;

    while (i <= r1 && j <= r2)
        res[k++] = (a[i] <= b[j]) ? a[i++] : b[j++];

    while (i <= r1) res[k++] = a[i++];
    while (j <= r2) res[k++] = b[j++];
}

typedef struct {
    int start;
    int end;
    double* a;
} sort_arg_t;

void* sort_thread(void* arg) {
    sort_arg_t* s = (sort_arg_t*)arg;
    insertion_sort(s->a + s->start, s->end - s->start);
    return NULL;
}

void sort_pthread(double* a, int n, int threads) {
    pthread_t th[threads];
    sort_arg_t args[threads];

    int chunk = n / threads;

    for (int t = 0; t < threads; t++) {
        args[t].start = t * chunk;
        args[t].end = (t == threads - 1) ? n : (t + 1) * chunk;
        args[t].a = a;
        pthread_create(&th[t], NULL, sort_thread, &args[t]);
    }

    for (int t = 0; t < threads; t++)
        pthread_join(th[t], NULL);

    /* последовательное слияние */
    double* tmp = malloc(sizeof(double) * n);
    int size = args[0].end;

    memcpy(tmp, a, sizeof(double) * size);

    for (int t = 1; t < threads; t++) {
        merge_two_sorted(
            tmp, 0, size - 1,
            a, args[t].start, args[t].end - 1,
            a, 0
        );
        size += args[t].end - args[t].start;
        memcpy(tmp, a, sizeof(double) * size);
    }

    free(tmp);
}

static inline unsigned int make_seed(int it, int i) {
    return 123456u + it * 100000u + i;
}

void generate(double* M1, double* M2, int N, int it) {

#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        unsigned int s = make_seed(it, i);
        M1[i] = 1.0 + ((double)rand_r(&s) / RAND_MAX) * (A - 1);
    }

#pragma omp parallel for
    for (int i = 0; i < N / 2; i++) {
        unsigned int s = make_seed(it, i + N);
        M2[i] = A + ((double)rand_r(&s) / RAND_MAX) * (9 * A);
    }
}

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


void merge(double* M1, double* M2, int N) {
#pragma omp parallel for
    for (int i = 0; i < N / 2; i++)
        M2[i] = fmin(M1[i], M2[i]);
}


double reduce(double* M2, int N) {
    double min = 0.0;

    for (int i = 0; i < N / 2; i++)
        if (M2[i] != 0.0) {
            min = M2[i];
            break;
        }

    double X = 0.0;
    if (min != 0.0) {
#pragma omp parallel for reduction(+:X)
        for (int i = 0; i < N / 2; i++) {
            int k = (int)(M2[i] / min);
            if (k % 2 == 0)
                X += sin(M2[i]);
        }
    }
    return X;
}

volatile int done = 0;


pthread_mutex_t print_mutex = PTHREAD_MUTEX_INITIALIZER;

void* progress_thread_fn(void* arg) {
    int total = *(int*)arg;
    while (1) {
        int current = done;
        
        pthread_mutex_lock(&print_mutex);
        printf("Progress: %.1f%%\n", 100.0 * (double)current / (double)total);
        pthread_mutex_unlock(&print_mutex);
        if (current >= total) {
            break;
        }
        sleep(1);
    }
    return NULL;
}

void progress(int total) {
    pthread_t th;
    int *parg = malloc(sizeof(int));
    if (!parg) return;
    *parg = total;
    pthread_create(&th, NULL, progress_thread_fn, parg);
    /* wait for progress thread to finish */
    pthread_join(th, NULL);
    free(parg);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s N THREADS\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int threads = atoi(argv[2]);

#ifdef _OPENMP
    omp_set_num_threads(threads);
    omp_set_max_active_levels(2);
#endif

    double* M1 = malloc(sizeof(double) * N);
    double* M2 = malloc(sizeof(double) * (N / 2));
    double* copy = malloc(sizeof(double) * (N / 2));

    double X = 0.0;

     double start_time = 0.0, end_time = 0.0;

  #pragma omp parallel num_threads(2) default(shared)
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        if (tid == 0) {
            progress(ITERATIONS);
        } else {

#ifdef _OPENMP
            omp_set_num_threads(threads);
#endif
            start_time = omp_get_wtime();
            for (int it = 0; it < ITERATIONS; it++) {
                generate(M1, M2, N, it + 1);
                map(M1, M2, copy, N);
                merge(M1, M2, N);

                sort_pthread(M2, N / 2, threads);
                X = reduce(M2, N);
#pragma omp atomic
                done++;
            }
            end_time = omp_get_wtime();
        }
    }

    double ms = (end_time - start_time) * 1000.0;

    {
        const char *outfname = "auto_output_lab5.txt";
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
    free(copy);
    return 0;
}
