#include "cuda.h"
#include "cuda_runtime.h"
#include <device_functions.h>
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"
#include <ctime>
#include <stdio.h>
#include "notbitwise.cuh"
#include "stdint.h"
#include "ErrorInfo.h"
#include "bitwise.cuh"
#include <ctime>
#include <iostream>
#include <string>

#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )


__global__ void initArrays(int combinations, int len, int realLength, bool* pop_bool, Data* pop_bw, int* pos, float* randomPC, int* randomPoint) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len) {
		pop_bool[idx] = false;
		pop_bool[idx + len] = true;
	}
	if (idx < realLength) {
		pop_bw[idx] = 0;
		pop_bw[idx + realLength] = ~((Data)0);
	}
	if (idx < combinations) {
		randomPC[idx] = 0;
		randomPoint[2 * idx] = idx / len;
		randomPoint[2 * idx + 1] = idx % len;
		pos[2 * idx] = 0;
		pos[2 * idx + 1] = 1;
	}
}

__global__ void comparar(Data* npop, bool* npop_bool, int* output, int length, int realLength) {
	Data aux;
	const int tid = threadIdx.x;
	__shared__ int intermedio[MAX_THREADS_PER_BLOCK];
	intermedio[tid] = 0;
	for (int i = tid;i < length;i = i + MAX_THREADS_PER_BLOCK) {
		int k = i / DataSize;
		int j = (DataSize - 1) - i % DataSize;

		aux = npop[blockIdx.x*realLength + k];
		bool value_bit = ((aux >> j) & 1) != 0;
		bool value_bool = npop_bool[blockIdx.x * length + i];
		if (value_bit != value_bool) {
			intermedio[tid] += 1;
		}
	}
	__syncthreads();
	for (unsigned int s = blockDim.x / 2; s != 0; s >>= 1) {
		if (tid < s) {
			intermedio[tid] = intermedio[tid] + intermedio[tid + s];
		}
		__syncthreads();
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		output[blockIdx.x] = intermedio[0];
	}



}

inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}

	return;
}

inline void __cudaCheckError(const char *file, const int line)
{

	cudaError err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}

	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	err = cudaDeviceSynchronize();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}

	return;
}

void PU_DPX() {
	Data *pop, *npop;
	bool *pop_bool, *npop_bool;

	Data *host_npop_bw;
	bool *host_npop_bool;

	float* randomPC;
	int *randomPoint, *pos, *output, *host_output;

	int largo = 64;
	int realLength = ((largo + DataSize - 1) / DataSize);
	int D = realLength * sizeof(Data);
	int output_count = 2 * largo * largo;
	cudaMalloc(&pop, 2 * D);
	CudaCheckError();
	cudaMalloc(&npop, output_count * D);
	CudaCheckError();
	cudaMalloc(&pop_bool, 2 * largo * sizeof(bool));
	cudaMalloc(&npop_bool, output_count * largo *  sizeof(bool));
	CudaCheckError();

	cudaMalloc(&pos, output_count * sizeof(int));
	CudaCheckError();
	cudaMalloc(&randomPoint, output_count * sizeof(int));
	CudaCheckError();
	cudaMalloc(&randomPC, output_count * sizeof(float));
	CudaCheckError();
	cudaMemset(randomPC, 0, output_count * sizeof(float));
	CudaCheckError();

	cudaMalloc(&output, output_count * sizeof(int));
	CudaCheckError();
	cudaMallocHost(&host_output, output_count * sizeof(int));
	CudaCheckError();

	initArrays << <largo, largo >> >(largo * largo, largo, realLength, pop_bool, pop, pos, randomPC, randomPoint);
	CudaCheckError();

	dpx << <output_count / 2, MAX_THREADS_PER_BLOCK >> >(pop_bool, npop_bool, pos, randomPC, randomPoint, largo, 1);
	CudaCheckError();
	dpx_b << <output_count / 2, MAX_THREADS_PER_BLOCK >> >(pop, npop, pos, randomPC, randomPoint, realLength, 1);
	CudaCheckError();
	comparar << <output_count, MAX_THREADS_PER_BLOCK >> >(npop, npop_bool, output, largo, realLength);
	CudaCheckError();
	cudaMemcpy(host_output, output, output_count, cudaMemcpyDeviceToHost);
	CudaCheckError();

	cudaMallocHost(&host_npop_bool, output_count * largo *  sizeof(bool));
	cudaMallocHost(&host_npop_bw, output_count * D);
	cudaMemcpy(host_npop_bool, npop_bool, output_count * largo *  sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_npop_bw, npop, output_count * D, cudaMemcpyDeviceToHost);

	for (int i = 0; i < output_count; i++) {
		if (host_output[i] > 0) {
			int pnt1 = (i / 2) / largo;
			int pnt2 = (i / 2) % largo;
			printf("%d: %d,%d[%d] -> err=%d\n", i, pnt1, pnt2, i % 2, host_output[i]);
			printf("bool: ");
			for (int j = 0; j < largo; j++) {
				printf("%d", host_npop_bool[i* largo + j] ? 1 : 0);
			}
			printf("\n");
			printf("bw  : ");
			for (int j = 0; j < largo; j++) {

				int k1 = j / DataSize;
				int k2 = (DataSize - 1) - j % DataSize;
				Data aux = host_npop_bw[i*realLength + k1];
				int value = (aux >> k2) & 1;
				printf("%d", value);
			}
			printf("\n");
		}
	}

}
