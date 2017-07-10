#include "cuda.h"
#include "cuda_runtime.h"
#include <device_functions.h>
#include "device_launch_parameters.h"
#include "curand.h"
#include <ctime>
#include <stdio.h>
#include "notbitwise.cuh"

#define MAX_THREADS_PER_BLOCK 1024


static const char *curandGetErrorString(curandStatus_t error)
{
	switch (error)
	{
	case CURAND_STATUS_SUCCESS:
		return "CURAND_STATUS_SUCCESS";

	case CURAND_STATUS_VERSION_MISMATCH:
		return "CURAND_STATUS_VERSION_MISMATCH";

	case CURAND_STATUS_NOT_INITIALIZED:
		return "CURAND_STATUS_NOT_INITIALIZED";

	case CURAND_STATUS_ALLOCATION_FAILED:
		return "CURAND_STATUS_ALLOCATION_FAILED";

	case CURAND_STATUS_TYPE_ERROR:
		return "CURAND_STATUS_TYPE_ERROR";

	case CURAND_STATUS_OUT_OF_RANGE:
		return "CURAND_STATUS_OUT_OF_RANGE";

	case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
		return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

	case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
		return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

	case CURAND_STATUS_LAUNCH_FAILURE:
		return "CURAND_STATUS_LAUNCH_FAILURE";

	case CURAND_STATUS_PREEXISTING_FAILURE:
		return "CURAND_STATUS_PREEXISTING_FAILURE";

	case CURAND_STATUS_INITIALIZATION_FAILED:
		return "CURAND_STATUS_INITIALIZATION_FAILED";

	case CURAND_STATUS_ARCH_MISMATCH:
		return "CURAND_STATUS_ARCH_MISMATCH";

	case CURAND_STATUS_INTERNAL_ERROR:
		return "CURAND_STATUS_INTERNAL_ERROR";
	}

	return "<unknown>";
}

struct ErrorInfo {
	cudaError_t cuda;
	curandStatus_t curand;

	ErrorInfo() : cuda(cudaSuccess), curand(CURAND_STATUS_SUCCESS){}

	bool ok() { return (cudaSuccess == this->cuda)  && (CURAND_STATUS_SUCCESS == this->curand); }
	bool failed() {
			if ((cudaSuccess == this->cuda) && (CURAND_STATUS_SUCCESS == this->curand)) {
				return false;
			}
			else {
				if (cudaSuccess != this->cuda) {
					printf("Error Cuda: %s\n", cudaGetErrorString(this->cuda));
				}
				if (CURAND_STATUS_SUCCESS != this->curand) {
					printf("Error Curand: %s (%d)\n", curandGetErrorString(this->curand),(int)this->curand);
				}
				return true;
			}
	}
};

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

__global__ void scaleRandom(float* floatRnd, int* intRnd, size_t N, unsigned int scale) {
	unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos < N) {
		intRnd[pos] = truncf(floatRnd[pos] * (scale + 0.999999f));
	}

}

ErrorInfo makeRandomIntegers(curandGenerator_t& generator, int* indices, unsigned int N, unsigned int max) {
	ErrorInfo status;
	float* rndFloat;

	status.cuda = cudaMalloc(&rndFloat, N * sizeof(float));
	if (status.failed()) return status;

	status.curand = curandGenerateUniform(generator, rndFloat, N);
	status.cuda = cudaDeviceSynchronize();
	if(status.failed()) return status;

	unsigned int blocks = (N + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK; // ceil(N/MAX_THREADS_PER_BLOCK)
	scaleRandom << <blocks, MAX_THREADS_PER_BLOCK >> >(rndFloat, indices, N, max);
	status.cuda = cudaGetLastError();
	if (status.failed()) return status;

	status.cuda = cudaDeviceSynchronize();

	cudaFree(rndFloat);
	return status;

}

curandStatus_t initGenerator(curandGenerator_t& generator) {
	curandStatus_t s =  curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_PHILOX4_32_10);
	if (s != CURAND_STATUS_SUCCESS) {
		return s; 
	}
	s = curandSetPseudoRandomGeneratorSeed(generator, clock());
	return s;
}

ErrorInfo initProbs(float** probs, int** points, size_t POP_SIZE) {
	ErrorInfo status;
	status.cuda = cudaMalloc(probs, POP_SIZE * sizeof(float));
	if (status.failed()) return status;
	status.cuda = cudaMalloc(points, POP_SIZE * sizeof(int));
	return status;
}

ErrorInfo makeRandomNumbersMutation(curandGenerator_t& generator, size_t POP_SIZE, int len, float* randomPM, int* randomPoint) {
	ErrorInfo status; 

	status.curand = curandGenerateUniform(generator, randomPM, POP_SIZE);
	if (status.failed()) return status;

	status = makeRandomIntegers(generator, randomPoint, POP_SIZE, len - 1);
	if (status.failed()) return status;
	status.cuda = cudaDeviceSynchronize();

	return status;

}



ErrorInfo makeRandomNumbersSpx(curandGenerator_t& generator, size_t POP_SIZE, int len, float* randomPC, int* randomPoint) {
	ErrorInfo status;
	size_t HALF_SIZE = POP_SIZE / 2;

	status = makeRandomIntegers(generator, randomPoint, HALF_SIZE, len - 1);
	if (status.failed()) return status;

	status.curand = curandGenerateUniform(generator, randomPC, HALF_SIZE);
	status.cuda = cudaDeviceSynchronize();

	return status;

}

ErrorInfo makeRandomNumbersDpx(curandGenerator_t& generator, size_t POP_SIZE, int len, float* randomPC, int* randomPoint) {
	ErrorInfo status;
	size_t HALF_SIZE = POP_SIZE / 2;

	status = makeRandomIntegers(generator, randomPoint, POP_SIZE, len - 1);
	if (status.failed()) return status;

	status.curand = curandGenerateUniform(generator, randomPC, HALF_SIZE);
	status.cuda = cudaDeviceSynchronize();

	return status;

}

cudaError_t InitTournRandom( int** random, size_t POP_SIZE) {
	return cudaMalloc(random, 2 * sizeof( int) * POP_SIZE);
}



ErrorInfo makeRandomNumbersTournement(curandGenerator_t& generator, size_t POP_SIZE, int* random) {
	// generar (POPSIZE * 2) numeros aleatorios enteros de 0 a POPSIZE - 1
	return makeRandomIntegers(generator, random, POP_SIZE * 2, POP_SIZE - 1);
}

cudaError_t InitFit(int** dev_fit,size_t POP_SIZE) {
	return cudaMalloc(dev_fit, sizeof(int) * POP_SIZE);
}
cudaError_t InitWin(int** dev_win, size_t POP_SIZE) {
	return cudaMalloc(dev_win, sizeof(int) * POP_SIZE );
}

ErrorInfo evaluate(bool* pop, size_t POP_SIZE, int length,int* dev_fit) {
	int *host_fit;
	double avgFit;
	int minFit, maxFit;
	ErrorInfo status;
	/*status.cuda = cudaMalloc(&dev_fit, sizeof(int) * POP_SIZE);
	if (status.failed()) return status;*/
	//status.cuda = cudaMallocHost((void**)&host_fit, sizeof(int) * POP_SIZE);
	if (status.failed()) return status;
	status.cuda = cudaDeviceSynchronize();
	if (status.failed()) return status;

	fitness <<< POP_SIZE, MAX_THREADS_PER_BLOCK >>> (pop, dev_fit, length);
	status.cuda = cudaGetLastError();
	if (status.failed()) return status;

	status.cuda = cudaDeviceSynchronize();
	if (status.failed()) return status;

	/*status.cuda = cudaMemcpy(host_fit, dev_fit, POP_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	if (status.failed()) return status;

	avgFit = maxFit = minFit  = host_fit[0];
	for (size_t i = 1; i < POP_SIZE; i++) {
		avgFit += host_fit[i];
		maxFit = host_fit[i] > maxFit ? host_fit[i] : maxFit;
		minFit = host_fit[i] < minFit ? host_fit[i] : minFit;
	}
	avgFit = (avgFit / POP_SIZE) / length;*/
	//printf("Min: %f, Max: %f, Avg: %f\n", minFit / (double)length,maxFit / (double) length, avgFit);
	//cudaFree(dev_fit);
	//cudaFreeHost(host_fit);
	return status;
}

__global__ void initPop(float* rnd, bool *pop,unsigned int length) {
	unsigned int idx = blockIdx.x;
	for (unsigned int i = threadIdx.x;i < length;i = i + MAX_THREADS_PER_BLOCK) {
		unsigned int pos = idx * length + i;
		pop[pos] = (rnd[pos] <= 0.5);
	}
}

ErrorInfo generatePOP(curandGenerator_t& generator, size_t POP_SIZE, int len, bool** pop,bool** npop) {
	ErrorInfo status;
	size_t N = POP_SIZE * len;
	float* dev_rnd;


	status.cuda = cudaMalloc(&dev_rnd, N * sizeof(float));
	if (status.failed()) return status;

	status.cuda = cudaMalloc(pop, N * sizeof(bool));
	if (status.failed()) return status;

	cudaMalloc(npop, N * sizeof(bool));
	if (status.failed()) return status;

	status.curand = curandGenerateUniform(generator, dev_rnd, N);
	status.cuda = cudaDeviceSynchronize();
	if (status.failed()) return status;

	initPop << < POP_SIZE, MAX_THREADS_PER_BLOCK >> >(dev_rnd, *pop, len);
	status.cuda = cudaGetLastError();
	if (status.failed()) return status;

	status.cuda = cudaDeviceSynchronize();
	cudaFree(dev_rnd);
	return status;
}

ErrorInfo GA(size_t POP_SIZE,int len,int iters,bool dpx_cross,float crossProb,float mutProb) {
	ErrorInfo status;
	bool *pop, *npop;
	int* fit;
	int* win;
	int* tourn;
	float  *probs;
	int *points;
	status.cuda = InitFit(&fit, POP_SIZE);
	status.cuda = InitWin(&win, POP_SIZE);
	status.cuda = InitTournRandom(&tourn, POP_SIZE);
	status = initProbs(&probs, &points, POP_SIZE);

	curandGenerator_t generator;
	status.curand = initGenerator(generator);
	if (status.failed()) return status;


	status = generatePOP(generator, POP_SIZE, len, &pop,&npop);
	

	if (status.failed()) {
		fprintf(stderr, "generatePOP failed!");
		return status;
	}
	//printf("gen %d: ", 0);
	status = evaluate(pop, POP_SIZE, len,fit);

	for (int gen = 1; gen <= iters; gen++) { // while not optimalSolutionFound
		// elegir POP_SIZE parejas para el torneo
		status = makeRandomNumbersTournement(generator, POP_SIZE, tourn);
		if (status.failed()) return status;

		// elegir POP_SIZE ganadores
		tournament<<< POP_SIZE / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK>>> (fit, tourn, win);
		status.cuda = cudaGetLastError();
		if (status.failed()) return status;

		// seleccion
		if (dpx_cross) {
			makeRandomNumbersDpx(generator, POP_SIZE, len, probs, points);
			dpx << < POP_SIZE / 2, MAX_THREADS_PER_BLOCK >> >(pop, npop, win, probs, points, len, crossProb);
		}
		else {
			makeRandomNumbersSpx(generator, POP_SIZE, len, probs, points);
			spx << < POP_SIZE / 2, MAX_THREADS_PER_BLOCK >> >(pop, npop, win, probs, points, len, crossProb);
		}
		status.cuda = cudaGetLastError();
		if (status.failed()) return status;

		status.cuda = cudaDeviceSynchronize();
		if (status.failed()) return status;

		// elegir numeros aleatorios para mutacion 
		// se reusa la memoria que se uso para los numeros aleatorios de la seleccion
		status = makeRandomNumbersMutation(generator, POP_SIZE, len, probs, points);

		// mutacion
		mutation << < POP_SIZE / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> >(npop, probs, points, len, mutProb);
		status.cuda = cudaGetLastError();
		if (status.failed()) return status;
		status.cuda = cudaDeviceSynchronize();

		bool* tmp;
		tmp = pop;
		pop = npop;
		npop = tmp;
		//printf("gen %d: ", gen);
		status = evaluate(pop, POP_SIZE, len, fit);
	}
	return status;
}


int main()
{


	cudaError_t cudaStatus;
	ErrorInfo status;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}


	unsigned int POP_SIZE = 2048 ;
	int len = 10000;
	
	GA(POP_SIZE, len, 1000, false, 0.9, 0.1);


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}


	return 0;
}