#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>


#define ITERATIONS 100


static inline unsigned int my_rand_r(unsigned int* seed) {
	*seed = (*seed * 1103515245u + 12345u);
	return (*seed / 65536u) % 32768u;
}


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


double get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec * 1e-6;
}


int main(int argc, char** argv) {
	if (argc < 3) {
		printf("Usage: %s N A\n", argv[0]);
		return 1;
	}


	int N = atoi(argv[1]);
	int A = atoi(argv[2]);


	double* M1 = malloc(sizeof(double) * N);
	double* M2 = malloc(sizeof(double) * (N / 2));


	double X = 0.0;


	for (int it = 0; it < ITERATIONS; it++) {
		unsigned int seed = it + 1;


		// Generate
		for (int i = 0; i < N; i++)
			M1[i] = 1.0 + (double)my_rand_r(&seed) / 32768.0 * A;


		for (int i = 0; i < N / 2; i++)
			M2[i] = A + (double)my_rand_r(&seed) / 32768.0 * 9 * A;


		// Map
		for (int i = 1; i < N / 2; i++)
			M2[i] = log(fabs(tan(M2[i] + M2[i - 1])));


		// Merge
		for (int i = 0; i < N / 2; i++)
			M2[i] = fmin(M1[i], M2[i]);


		// Sort
		insertion_sort(M2, N / 2);


		// Reduce
		double min = 0.0;
		for (int i = 0; i < N / 2; i++)
			if (M2[i] != 0.0) {
				min = M2[i];
				break;
			}


		X = 0.0;
		for (int i = 0; i < N / 2; i++) {
			if (min != 0.0) {
				int k = (int)(M2[i] / min);
				if (k % 2 == 0)
					X += sin(M2[i]);
			}
		}
	}


	printf("X = %.10f\n", X);


	free(M1);
	free(M2);
	return 0;
}