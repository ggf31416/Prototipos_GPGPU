#include "cuda.h"
#include "cuda_runtime.h"
#include <device_functions.h>
#include "device_launch_parameters.h"
#include "curand.h"
#include <ctime>
#include <stdio.h>





__global__ void reduction(float* dev_rnd, float* dev_output)
{
	extern __shared__ float intermedio[];
	unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
	unsigned int tid = threadIdx.x;
	intermedio[threadIdx.x] = dev_rnd[idx] + dev_rnd[idx + blockDim.x];
	__syncthreads();
	for (unsigned int s = blockDim.x / 2; s != 0; s >>= 1) {
		if (tid < s) {
			intermedio[tid] += intermedio[tid + s];
		}
		__syncthreads();
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		dev_output[blockIdx.x] = intermedio[0];
	}

	
}

int RandomHostAPI_Test(int numBlocks,int numThreads) {
	float* dev_rnd;
	float* dev_output;
	cudaError_t cudaStatus; 
	curandGenerator_t generator;
	curandStatus_t s  = curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_XORWOW);
	if (s != CURAND_STATUS_SUCCESS) {
		return s; // cualquiera != 0;
	}
	s = curandSetPseudoRandomGeneratorSeed(generator, clock());
	int N = numBlocks * numThreads;
	cudaStatus = cudaMalloc((void**)&dev_rnd, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}
	cudaStatus = cudaMalloc((void**)&dev_output, numBlocks * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}
	s = curandGenerateUniform(generator, dev_rnd, N);
	if (s != CURAND_STATUS_SUCCESS) {
		return s; // cualquiera != 0;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}
	s = curandDestroyGenerator(generator);
	if (s != CURAND_STATUS_SUCCESS) {
		return s; // cualquiera != 0;
	}
	reduction << < numBlocks / 2, numThreads, numThreads * sizeof(float) >> > (dev_rnd, dev_output);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}
	float* host_output;
	cudaStatus = cudaMallocHost(&host_output, numBlocks * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(host_output, dev_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}
	float suma = 0;
	for (int i = 0; i < numBlocks; i++) {
		suma += host_output[i];
	}
	printf("Sum: %f", suma * 2);

	cudaFreeHost(host_output);
	cudaFree(dev_output);
	cudaFree(dev_rnd);
	return cudaStatus;
}

int main()
{


	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

    cudaStatus = (cudaError_t) RandomHostAPI_Test(100000,1024);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }


    return 0;
}
