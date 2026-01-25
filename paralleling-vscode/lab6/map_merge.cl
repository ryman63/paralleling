#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Kernel 1: вычисление M1 (векторная операция)
__kernel void compute_M1(__global double* M1, int N) {
    int i = get_global_id(0);
    if (i < N) {
        M1[i] = exp(sqrt(M1[i]));
    }
}

// Kernel 2: вычисление M2 с последовательной зависимостью
__kernel void compute_M2(__global double* M2, __global double* copy, int N) {
    int i = get_global_id(0);
    
    // Копирование в copy
    if (i < N / 2) {
        copy[i] = M2[i];
    }
    
    // Важно: глобальный барьер через запуск kernel
    // (гарантируется, что все копии завершены перед вторым проходом)
}

// Kernel 3: вычисление M2 с использованием copy[i-1]
__kernel void compute_M2_final(__global double* M2, __global double* copy, int N) {
    int i = get_global_id(0);
    if (i < N / 2) {
        double prev = (i == 0) ? 0.0 : copy[i - 1];
        M2[i] = fabs(tan(copy[i] + prev));
    }
}

// Kernel 4: merge M1 и M2
__kernel void merge(__global double* M1, __global double* M2, int N) {
    int i = get_global_id(0);
    if (i < N / 2) {
        M2[i] = fmin(M1[i], M2[i]);
    }
}