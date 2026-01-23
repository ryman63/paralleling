#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>
#include <omp.h>

#define ITERATIONS 100
#define A 504

/* ================= CPU SORT ================= */

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

/* ================= DATA ================= */

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

    /* ===== OpenCL init ===== */

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        printf("clGetPlatformIDs error %d\n", err);
        return 1;
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf("clGetDeviceIDs error %d\n", err);
        return 1;
    }

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("clCreateContext error %d\n", err);
        return 1;
    }

    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        printf("clCreateCommandQueue error %d\n", err);
        return 1;
    }

    /* ===== Load kernel ===== */

    FILE* f = fopen("map_merge.cl", "rb");
    if (!f) {
        perror("map_merge.cl");
        return 1;
    }

    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    rewind(f);

    char* src = malloc(sz + 1);
    fread(src, 1, sz, f);
    src[sz] = 0;
    fclose(f);

    program = clCreateProgramWithSource(context, 1,
        (const char**)&src, &sz, &err);

    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device,
            CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        char* log = malloc(log_size + 1);
        clGetProgramBuildInfo(program, device,
            CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = 0;

        printf("BUILD LOG:\n%s\n", log);
        return 1;
    }

    kernel = clCreateKernel(program, "map_merge", &err);
    if (err != CL_SUCCESS) {
        printf("clCreateKernel error %d\n", err);
        return 1;
    }

    /* ===== Buffers ===== */

    cl_mem d_M1 = clCreateBuffer(context, CL_MEM_READ_ONLY,
        sizeof(double) * N, NULL, NULL);
    cl_mem d_M2 = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(double) * (N / 2), NULL, NULL);

    /* ===== Progress thread ===== */

    pthread_t pth;
    int total_iters = ITERATIONS;
    pthread_create(&pth, NULL, progress_thread, &total_iters);

    double start_time = omp_get_wtime();

    double X = 0.0;
    for (int it = 0; it < ITERATIONS; it++) {
        generate(M1, M2, N, it + 1);

        clEnqueueWriteBuffer(queue, d_M1, CL_TRUE, 0,
            sizeof(double) * N, M1, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, d_M2, CL_TRUE, 0,
            sizeof(double) * (N / 2), M2, 0, NULL, NULL);

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_M1);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_M2);
        clSetKernelArg(kernel, 2, sizeof(int), &N);

        size_t global = N / 2;
        clEnqueueNDRangeKernel(queue, kernel, 1,
            NULL, &global, NULL, 0, NULL, NULL);
        clFinish(queue);

        clEnqueueReadBuffer(queue, d_M2, CL_TRUE, 0,
            sizeof(double) * (N / 2), M2, 0, NULL, NULL);

        sort_pthread(M2, N / 2, threads);
        X = reduce(M2, N);
        done++;
    }

    double ms = (omp_get_wtime() - start_time) * 1000.0;
    pthread_join(pth, NULL);

    { const char *outfname = "auto_output_lab6.txt"; 
        FILE *out = fopen(outfname, "w"); 
        if (out) { fprintf(out, "%d %d %.3f %.10f\n", N, threads, ms, X); 
            fclose(out); 
        } 
        else { 
            fprintf(stderr, "Warning: cannot open %s for writing\n", outfname); 
        } 
    }

    printf("Result X = %.10f\n", X);
    printf("Time: %.3f ms\n", ms);

    return 0;
}
