#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

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

void generate(double* M1, double* M2, int N, unsigned int* seed) {
    // M1: [1, A]
    for (int i = 0; i < N; i++) {
        M1[i] = 1.0 + ((double)rand_r(seed) / RAND_MAX) * (A - 1);
    }

    // M2: [A, 10*A]
    for (int i = 0; i < N / 2; i++) {
        M2[i] = A + ((double)rand_r(seed) / RAND_MAX) * (9 * A);
    }
}

void map(double* M1, double* M2, int N) {
    // M1: exp(sqrt(x))
    for (int i = 0; i < N; i++) {
        M1[i] = exp(sqrt(M1[i]));
    }

    // M2: |tan(M2[i] + M2[i-1])|
    double* M2_copy = malloc(sizeof(double) * (N / 2));
    for (int i = 0; i < N / 2; i++) {
        M2_copy[i] = M2[i];
    }

    for (int i = 0; i < N / 2; i++) {
        double prev = (i == 0) ? 0.0 : M2_copy[i - 1];
        M2[i] = fabs(tan(M2_copy[i] + prev));
    }

    free(M2_copy);
}

void merge(double* M1, double* M2, int N) {
    for (int i = 0; i < N / 2; i++) {
        M2[i] = fmin(M1[i], M2[i]);
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
            if (k % 2 == 0) {
                X += sin(M2[i]);
            }
        }
    }

    return X;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s N\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);

    double* M1 = malloc(sizeof(double) * N);
    double* M2 = malloc(sizeof(double) * (N / 2));

    struct timeval T1, T2;
    double X = 0.0;

    gettimeofday(&T1, NULL);

    for (int it = 0; it < ITERATIONS; it++) {
        unsigned int seed = it + 1;

        generate(M1, M2, N, &seed);
        map(M1, M2, N);
        merge(M1, M2, N);
        insertion_sort(M2, N / 2);
        X = reduce(M2, N);
    }

    gettimeofday(&T2, NULL);

    float delta_ms = (T2.tv_sec - T1.tv_sec) * 1000.0f +
        (T2.tv_usec - T1.tv_usec) / 1000.0f;

    printf("%d %.2f %.10f\n", N, delta_ms, X);

    free(M1);
    free(M2);
    return 0;
}