#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void map_merge(
    __global double* M1,
    __global double* M2,
    __global double* copy,
    int N
) {
    int i = get_global_id(0);

    if (i < N) {
        M1[i] = exp(sqrt(M1[i]));
    }

    if (i < N / 2) {
        copy[i] = M2[i];
        barrier(CLK_GLOBAL_MEM_FENCE);

        double prev = (i == 0) ? 0.0 : copy[i - 1];
        M2[i] = fabs(tan(copy[i] + prev));

        M2[i] = fmin(M1[i], M2[i]);
    }
}
