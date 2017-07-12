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

#define SALIDA 1
#define SALIDA_STEP 50
#define INIT_THREADS 128

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

struct EvalInfo {
	double min;
	double max;
	double avg;
};

template<typename T>
__global__ void sumar(T* dev_rnd, float* dev_output,unsigned int len)
{
	__shared__ float intermedio[MAX_THREADS_PER_BLOCK];
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tid = threadIdx.x;
	intermedio[threadIdx.x] = idx < len ?  dev_rnd[idx] : 0;
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

// sumar mas optimizado pero mas rigido
__global__ void sumar2(float* dev_rnd, float* dev_output)
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

__global__ void minimo(int* dev_rnd, int* dev_output,unsigned int len)
{
	__shared__ int intermedio[MAX_THREADS_PER_BLOCK];
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tid = threadIdx.x;
	intermedio[threadIdx.x] = idx < len ? dev_rnd[idx] : dev_rnd[0];
	__syncthreads();
	for (unsigned int s = blockDim.x / 2; s != 0; s >>= 1) {
		if (tid < s) {
			intermedio[tid] = min(intermedio[tid], intermedio[tid + s]);
		}
		__syncthreads();
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		dev_output[blockIdx.x] = intermedio[0];
	}
}

__global__ void maximo(int* dev_rnd, int* dev_output, unsigned int len)
{
	__shared__ int intermedio[MAX_THREADS_PER_BLOCK];
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tid = threadIdx.x;
	intermedio[threadIdx.x] = idx < len ? dev_rnd[idx] : dev_rnd[0];
	__syncthreads();
	for (unsigned int s = blockDim.x / 2; s != 0; s >>= 1) {
		if (tid < s) {
			intermedio[tid] = max(intermedio[tid], intermedio[tid + s]);
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

curandStatus_t initGenerator(curandGenerator_t& generator ,unsigned long long seed) {
	curandStatus_t s =  curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_PHILOX4_32_10);
	if (s != CURAND_STATUS_SUCCESS) {
		return s; 
	}
	s = curandSetPseudoRandomGeneratorSeed(generator,seed);
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


ErrorInfo evaluate(bool* pop, size_t POP_SIZE, int length,int* dev_fit,EvalInfo& eval) {

	float avgFit;
	int minFit, maxFit;
	ErrorInfo status;



	fitness <<< POP_SIZE, MAX_THREADS_PER_BLOCK >>> (pop, dev_fit, length);
	status.cuda = cudaGetLastError();
	if (status.failed()) return status;

	status.cuda = cudaDeviceSynchronize();
	if (status.failed()) return status;


	int* out1;
	float* out2;
	
	int nroBlocks = (POP_SIZE  + MAX_THREADS_PER_BLOCK - 1) / (MAX_THREADS_PER_BLOCK);
	cudaMalloc(&out1, nroBlocks * sizeof(int));
	cudaMalloc(&out2, nroBlocks * sizeof(float));
	//minimo << <nroBlocks, MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK * sizeof(float) >> >(dev_fit, out1);
	minimo << <nroBlocks, MAX_THREADS_PER_BLOCK >> >(dev_fit, out1,POP_SIZE);
	minimo << <1, MAX_THREADS_PER_BLOCK >> >(out1, out1, nroBlocks);
	cudaMemcpy(&minFit, out1, sizeof(int), cudaMemcpyDeviceToHost);


	maximo << <nroBlocks, MAX_THREADS_PER_BLOCK >> >(dev_fit, out1, POP_SIZE);
	maximo << <1, MAX_THREADS_PER_BLOCK >> >(out1, out1, nroBlocks);
	cudaMemcpy(&maxFit, out1, sizeof(int), cudaMemcpyDeviceToHost);

	sumar << <nroBlocks, MAX_THREADS_PER_BLOCK >> >(dev_fit, out2, POP_SIZE);
	sumar << <1, MAX_THREADS_PER_BLOCK >> >(out2, out2, nroBlocks);
	cudaMemcpy(&avgFit, out2, sizeof(float), cudaMemcpyDeviceToHost);

	/*
	int *host_fit;
	status.cuda = cudaMallocHost((void**)&host_fit, sizeof(int) * POP_SIZE); // para calcular max,min,avg
	status.cuda = cudaMemcpy(host_fit, dev_fit, POP_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	if (status.failed()) return status;

	avgFit = maxFit = minFit  = host_fit[0];
	for (size_t i = 1; i < POP_SIZE; i++) {
		avgFit += host_fit[i];
		maxFit = host_fit[i] > maxFit ? host_fit[i] : maxFit;
		minFit = host_fit[i] < minFit ? host_fit[i] : minFit;
	}
	//cudaFreeHost(host_fit);
	*/
	cudaFree(out1);
	cudaFree(out2);

	avgFit = (avgFit / POP_SIZE) / length;
	eval.min = minFit / (double)length;
	eval.max = maxFit / (double)length;
	eval.avg = avgFit;
	//if (SALIDA) printf("Min: %f, Max: %f, Avg: %f\n", minFit / (double)length,maxFit / (double) length, avgFit);
	
	return status;
}


// Thamas Wang
// http://www.burtleburtle.net/bob/hash/integer.html
uint64_t hash64shift(uint64_t key)
{
	key = (~key) + (key << 21); // key = (key << 21) - key - 1;
	key = key ^ (key >>  24);
	key = (key + (key << 3)) + (key << 8); // key * 265
	key = key ^ (key >>  14);
	key = (key + (key << 2)) + (key << 4); // key * 21
	key = key ^ (key >>  28);
	key = key + (key << 31);
	return key;
}

__global__ void initPop_device(bool *pop, unsigned int length,unsigned long long seed) {
	unsigned int thIdx = blockIdx.x * blockDim.x + threadIdx.x;
	curandStatePhilox4_32_10_t rndState;
	curand_init(seed + thIdx, 0ull, 0ull, &rndState);
	for (unsigned int i = threadIdx.x; i < length; i = i + INIT_THREADS) {
		unsigned int pos = blockIdx.x * length + i;
		pop[pos] = (curand_uniform(&rndState) <= 0.5);
	}
}

__global__ void initPop_device32(bool *pop, unsigned int length, unsigned long long seed) {
	unsigned int thIdx = blockIdx.x * blockDim.x + threadIdx.x;
	curandStatePhilox4_32_10_t rndState;
	curand_init(seed + thIdx, 0ull, 0ull, &rndState);
	for (unsigned int i = threadIdx.x; i < length; ) {
		uint32_t rnd = curand(&rndState);
		for (uint32_t j = 0; j < 32 & i < length; j++, i = i + INIT_THREADS) {
			unsigned int pos = blockIdx.x * length + i;
			pop[pos] = (rnd & (1 << j)) != 0;
		}

	}
}

ErrorInfo generatePOP_device(unsigned long seed, size_t POP_SIZE, int len, bool** pop, bool** npop) {
	ErrorInfo status;
	size_t N = POP_SIZE * len;
	status.cuda = cudaMalloc(pop, N * sizeof(bool));
	if (status.failed()) return status;
	status.cuda = cudaMalloc(npop, N * sizeof(bool));
	if (status.failed()) return status;

	initPop_device32 << < POP_SIZE, INIT_THREADS >> >( *pop, len,seed);
	status.cuda = cudaGetLastError();
	if (status.failed()) return status;

	status.cuda = cudaDeviceSynchronize();
	return status;
}

__global__ void initPop(float* rnd, bool *pop,unsigned int length) {
	unsigned int idx = blockIdx.x;
	for (unsigned int i = threadIdx.x; i < length; i = i + MAX_THREADS_PER_BLOCK) {
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

	status.cuda = cudaMalloc(npop, N * sizeof(bool));
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
	unsigned long long seed = 42;
	ErrorInfo status;
	bool *pop, *npop;
	int* fit;
	int* win;
	int* tourn;
	float  *probs;
	int *points;
	EvalInfo eval;
	status.cuda = InitFit(&fit, POP_SIZE);
	status.cuda = InitWin(&win, POP_SIZE);
	status.cuda = InitTournRandom(&tourn, POP_SIZE);
	status = initProbs(&probs, &points, POP_SIZE);

	curandGenerator_t generator;
	status.curand = initGenerator(generator, seed);
	if (status.failed()) return status;


	//status = generatePOP(generator, POP_SIZE, len, &pop,&npop);
	// usa la curand device API para generar la poblacion sin prealocar numeros aleatorios para eso
	status = generatePOP_device(seed, POP_SIZE, len, &pop, &npop);
	//status = generatePOP_device(hash64shift(seed), POP_SIZE, len, &pop, &npop);

	// cambia el offset del generador para que no se sobreponga con el usado para la generacion de la poblacion
	curandSetGeneratorOffset(generator, POP_SIZE * len);

	if (status.failed()) {
		fprintf(stderr, "generatePOP failed!");
		return status;
	}

	
	status = evaluate(pop, POP_SIZE, len,fit,eval);
	if (SALIDA) printf("gen %d: Min: %f, Max: %f, Avg: %f\n", 0, eval.min, eval.max, eval.avg);

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
		status = evaluate(pop, POP_SIZE, len, fit,eval);
		if (SALIDA && (gen % SALIDA_STEP) == 0) printf("gen %d: Min: %f, Max: %f, Avg: %f\n", gen, eval.min, eval.max, eval.avg);
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

	// TODO: arreglar para POP_SIZE no multiplo de MAX_THREADS
	unsigned int POP_SIZE = 2048 ;
	int len = 10000;
	int iters = 3000; 
	float pMutacion = 0.4;
	float pCruce = 1;
	
	GA(POP_SIZE, len, iters, false, pCruce,pMutacion);


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}


	return 0;
}
