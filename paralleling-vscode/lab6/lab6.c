#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

#define ITERATIONS 100
#define A 504

/* ================= SORT (pthread) ================= */

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
}

/* ================= DATA GENERATION ================= */

static inline unsigned int make_seed(int it, int i) {
    return 123456u + it * 100000u + i;
}

void generate(double* M1, double* M2, int N, int it) {
    for (int i = 0; i < N; i++) {
        unsigned int s = make_seed(it, i);
        M1[i] = 1.0 + ((double)rand_r(&s) / RAND_MAX) * (A - 1);
    }

    for (int i = 0; i < N / 2; i++) {
        unsigned int s = make_seed(it, i + N);
        M2[i] = A + ((double)rand_r(&s) / RAND_MAX) * (9 * A);
    }
}

/* ================= REDUCE ================= */

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
        for (int i = 0; i < N / 2; i++) {
            int k = (int)(M2[i] / min);
            if (k % 2 == 0)
                X += sin(M2[i]);
        }
    }
    return X;
}

/* ================= PROGRESS ================= */

volatile int done = 0;
pthread_mutex_t print_mutex = PTHREAD_MUTEX_INITIALIZER;

void* progress_thread(void* arg) {
    int total = *(int*)arg;
    while (1) {
        pthread_mutex_lock(&print_mutex);
        printf("Progress: %.1f%%\n", 100.0 * done / total);
        pthread_mutex_unlock(&print_mutex);

        if (done >= total) break;
        sleep(1);
    }
    return NULL;
}

/* ================= MAIN ================= */

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s N THREADS\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int threads = atoi(argv[2]);

    double* M1 = malloc(sizeof(double) * N);
    double* M2 = malloc(sizeof(double) * (N / 2));
    double* copy = malloc(sizeof(double) * (N / 2));

    /* ===== OpenCL init ===== */

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device, 0, NULL);

    FILE* f = fopen("map_merge.cl", "rb");
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    rewind(f);
    char* src = malloc(sz + 1);
    fread(src, 1, sz, f);
    src[sz] = 0;
    fclose(f);

    program = clCreateProgramWithSource(context, 1, (const char**)&src, &sz, NULL);
    clBuildProgram(program, 1, &device, "", NULL, NULL);
    kernel = clCreateKernel(program, "map_merge", NULL);

    cl_mem d_M1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * N, NULL, NULL);
    cl_mem d_M2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * (N / 2), NULL, NULL);
    cl_mem d_copy = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * (N / 2), NULL, NULL);

    pthread_t pth;
    pthread_create(&pth, NULL, progress_thread, &((int){ITERATIONS}));

    double start = omp_get_wtime();

    double X = 0.0;
    for (int it = 0; it < ITERATIONS; it++) {
        generate(M1, M2, N, it + 1);

        clEnqueueWriteBuffer(queue, d_M1, CL_TRUE, 0, sizeof(double) * N, M1, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, d_M2, CL_TRUE, 0, sizeof(double) * (N / 2), M2, 0, NULL, NULL);

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_M1);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_M2);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_copy);
        clSetKernelArg(kernel, 3, sizeof(int), &N);

        size_t global = N;
        clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
        clFinish(queue);

        clEnqueueReadBuffer(queue, d_M2, CL_TRUE, 0,
            sizeof(double) * (N / 2), M2, 0, NULL, NULL);

        sort_pthread(M2, N / 2, threads);
        X = reduce(M2, N);
        done++;
    }

    double end = omp_get_wtime();
    pthread_join(pth, NULL);

    printf("Result X = %.10f\n", X);
    printf("Time: %.3f ms\n", (end - start) * 1000.0);

    return 0;
}
