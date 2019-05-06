#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>
using namespace std;

// function generate random numbers and assign it to array
void random_ints(int *a, int N)
{
	for (int i = 0; i < N; i++)
		a[i] = rand();
}

// create kernal "two vector addition"
__global__ void add(int *a, int *b, int *c) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	c[index] = a[index] + b[index];
}

#define N 8                      // array size 
#define THREADS_PER_BLOCK 2

int main(void) {

	int *a, *b, *c;              // host data 
	int *d_a, *d_b, *d_c;		 // device data
	int size = N * sizeof(int);

	cout<<"\t\t\t*** CUDA TASK ***\n\t\t\t-----------------\n\n";

	// alloacate host data 
	a = (int *)malloc(size); random_ints(a, N);
	b = (int *)malloc(size); random_ints(b, N);
	c = (int *)malloc(size);

	// allocate device data
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	// copy data from host to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	// call add kernal 
	add << < N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_a, d_b, d_c);

	// copy data back from device to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	cout<<"A\tB\tC\n------------------------\n";

	for (int i = 0; i < N; i++) {
		cout<<a[i]<<"\t"<<b[i]<<"\t"<<c[i]<<"\n";
		cout<<"------------------------\n";
	}

	// free allocated host data
	free(a);  free(b);  free(c);

	// free allocated device data 
	cudaFree(d_a);  cudaFree(d_b);  cudaFree(d_c);

	return 0;
}
