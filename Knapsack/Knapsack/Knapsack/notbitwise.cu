
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "curand_kernel.h"
#include "notbitwise.cuh"
#define V2

#define KNAPSACK
// Kernel invocation: mutation <<<N_BLOCK,BLOCK_LENGTH>>> (pgpu,randomPM,randomPoint,CHROM_LEN,PROB_MUT);
// N_BLOCK: number of blocks, such that N_BLOCK * BLOCK_LENGTH = POP_SIZE.
// BLOCK_LENGTH: length of each block, such that N_BLOCK * BLOCK_LENGTH = POP_SIZE.
// pgpu: pointer to global memory that stores the population. 
// randomPM: pointer to global memory that stores the random values for mutation.
// randomPoint: pointer to global memory that stores the random points for mutation.
// CHROM_LEN: length of the chromosome.
// PROB_MUT: mutation probability. 

__global__ void mutation(bool *pop, float *randomPM, int *randomPoint, int length, float PROB_MUT) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float pm = randomPM[idx];   // value for mutation
	int pnt = randomPoint[idx]; // mutation point

	if (pm <= PROB_MUT) {
		pop[idx*length + pnt] = !pop[idx*length + pnt];
	}

}



__device__ inline void warpReduce2(volatile float * partial, const unsigned int tid) {
	if (MAX_THREADS_PER_BLOCK >= 64) {
		partial[tid] = partial[tid] + partial[tid + 32];
	}

	if (MAX_THREADS_PER_BLOCK >= 32) {
		partial[tid] = partial[tid] + partial[tid + 16];
	}

	if (MAX_THREADS_PER_BLOCK >= 16) {
		partial[tid] = partial[tid] + partial[tid + 8];
	}

	if (MAX_THREADS_PER_BLOCK >= 8) {
		partial[tid] = partial[tid] + partial[tid + 4];
	}

	if (MAX_THREADS_PER_BLOCK >= 4) {
		partial[tid] = partial[tid] + partial[tid + 2];
	}

	if (MAX_THREADS_PER_BLOCK >= 2) {
		partial[tid] = partial[tid] + partial[tid + 1];
	}
}


// Kernel invocation: fitness <<<POP_SIZE,MAX_THREADS_PER_BLOCK>>> (pgpu,fitgpu,CHROM_LEN);
// POP_SIZE: number of individuals of the population.
// MAX_THREADS_PER_BLOCK: constant with the maximum number of threads per block. 
// CHROM_LEN: length of the chromosome.
// pgpu: pointer to global memory that stores the population. 

__global__ void fitness_knapsack(bool *pop, float* fit, int length, float* W, float* G, float MAX_W,float P) {
	__shared__ float partial_g[MAX_THREADS_PER_BLOCK];  // array to store partial gain
	__shared__ float partial_w[MAX_THREADS_PER_BLOCK];  // array to store partial weights
	int idx = blockIdx.x;  // the number of the individual is indicated by the block number
	const unsigned int tid = threadIdx.x;
	partial_g[tid] = 0; // each array position is initialized in 0
	partial_w[tid] = 0; // each array position is initialized in 0

	int i;
	// each thread adds in partial its corresponding values
	for (i = tid;i<length;i = i + MAX_THREADS_PER_BLOCK) {
		float w = pop[idx*length + i] * W[i];
		float g = pop[idx*length + i] * G[i];
		partial_g[tid] = partial_g[tid] + g;
		partial_w[tid] = partial_w[tid] + w;
	}

	// reduction algorithm to add the partial fitness values
	__syncthreads();

	if (MAX_THREADS_PER_BLOCK >= 1024) {
		if (tid < 512) {
			partial_w[tid] = partial_w[tid] + partial_w[tid + 512];
			partial_g[tid] = partial_g[tid] + partial_g[tid + 512];
		}
		__syncthreads();
	}
	if (MAX_THREADS_PER_BLOCK >= 512) {
		if (tid < 256) {
			partial_w[tid] = partial_w[tid] + partial_w[tid + 256];
			partial_g[tid] = partial_g[tid] + partial_g[tid + 256];
		}
		__syncthreads();
	}
	if (MAX_THREADS_PER_BLOCK >= 256) {
		if (tid < 128) {
			partial_w[tid] = partial_w[tid] + partial_w[tid + 128];
			partial_g[tid] = partial_g[tid] + partial_g[tid + 128];
		}
		__syncthreads();
	}
	if (MAX_THREADS_PER_BLOCK >= 128) {
		if (tid < 64) {
			partial_w[tid] = partial_w[tid] + partial_w[tid + 64];
			partial_g[tid] = partial_g[tid] + partial_g[tid + 64];
		}
		__syncthreads();
	}
	if (tid < 32) {
		warpReduce2(partial_w, tid);
		warpReduce2(partial_g, tid);
		// finally thread 0 writes the fitness value in global memory
		if (tid == 0) {

			fit[idx] = partial_w[0] <= MAX_W ? partial_g[0] : -P * (partial_w[0] - MAX_W);
			/*if (blockIdx.x == 0) {
				printf("g: %f w: %f fit:%f\n", partial_g[0], partial_w[0], fit[idx]);
			}*/
		}
	}
}

// Kernel invocation: dpx <<POP_SIZE/2,MAX_THREADS_PER_BLOCK>>> (oldpop,newpop,positions,randomPC,randomPoint,
// CHROM_LEN,PROB_CROSS); 
// POP_SIZE: number of individuals of the population.
// MAX_THREADS_PER_BLOCK: constant with the maximum number of threads per block. 
// oldpop: pointer to global memory that stores the old population. 
// newpop: pointer to global memory that stores the new population.
// positions: pointer to global memory that stores the indexes of individuals for crossover
// randomPC: pointer to global memory that stores the random values for crossover.
// randomPoint: pointer to global memory that stores the random points for crossover.
// CHROM_LEN: length of the chromosome.
// PROB_CROSS: crossover probability.

__global__ void dpx(bool *pop, bool *npop, int *pos, float *randomPC, int *randomPoint, int length, float PROB_CROSS) {

	int idx = blockIdx.x;  // the number of the individual is indicated by the block number
	int ind1 = pos[2 * idx];  // index of first individual for crossover
	int ind2 = pos[2 * idx + 1]; // index of second individual for crossover
	float pc = randomPC[idx]; // value for crossover
	int pnt1 = randomPoint[2 * idx]; // first crossover point
	int pnt2 = randomPoint[2 * idx + 1]; // second crossover point
	int i;

	if (pc <= PROB_CROSS) {
		// Cross individuals
		for (i = threadIdx.x;i<length;i = i + MAX_THREADS_PER_BLOCK) {
			if (i<pnt1 || i >= pnt2) {
				//copy bit from parent ind1 to child 2*idx
				npop[2 * idx*length + i] = pop[ind1*length + i];
				//copy bit from parent ind2 to child 2*idx + 1
				npop[(2 * idx + 1)*length + i] = pop[ind2*length + i];
			}
			else {
				//copy bit from parent ind2 to child 2*idx	       		    
				npop[2 * idx*length + i] = pop[ind2*length + i];
				//copy bit from parent ind1 to child 2*idx + 1
				npop[(2 * idx + 1)*length + i] = pop[ind1*length + i];
			}
		}
	}
	else {
		// Copy individuals
		for (i = threadIdx.x;i<length;i = i + MAX_THREADS_PER_BLOCK) {
			//copy bit from parent ind1 to child 2*idx
			npop[(2 * idx)*length + i] = pop[ind1*length + i];
			//copy bit from parent ind2 to child 2*idx + 1
			npop[(2 * idx + 1)*length + i] = pop[ind2*length + i];
		}
	}

}



// Kernel invocation: spx <<POP_SIZE/2,MAX_THREADS_PER_BLOCK>>> (oldpop,newpop,positions,randomPC,randomPoint,
// CHROM_LEN,PROB_CROSS); 
// POP_SIZE: number of individuals of the population.
// MAX_THREADS_PER_BLOCK: constant with the maximum number of threads per block. 
// oldpop: pointer to global memory that stores the old population. 
// newpop: pointer to global memory that stores the new population.
// positions: pointer to global memory that stores the indexes of individuals for crossover
// randomPC: pointer to global memory that stores the random values for crossover.
// randomPoint: pointer to global memory that stores the random points for crossover.
// CHROM_LEN: length of the chromosome.
// PROB_CROSS: crossover probability. 

__global__ void spx(bool *pop, bool *npop, int *pos, float *randomPC, int *randomPoint, int length, float PROB_CROSS) {

	int idx = blockIdx.x; // the number of the individual is indicated by the block number
	int ind1 = pos[2 * idx]; // index of first individual for crossover
	int ind2 = pos[2 * idx + 1]; // index of second individual for crossover
	float pc = randomPC[idx]; // value for crossover
	int pnt = randomPoint[idx]; // crossover point
	int i;

	if (pc <= PROB_CROSS) {
		// Cross individuals
		for (i = threadIdx.x;i<length;i = i + MAX_THREADS_PER_BLOCK) {
			if (i<pnt) {
				//copy bit from parent ind1 to child 2*idx
				npop[2 * idx*length + i] = pop[ind1*length + i];
				//copy bit from parent ind2 to child 2*idx + 1
				npop[(2 * idx + 1)*length + i] = pop[ind2*length + i];
			}
			else {
				//copy bit from parent ind2 to child 2*idx
				npop[2 * idx*length + i] = pop[ind2*length + i];
				//copy bit from parent ind1 to child 2*idx + 1
				npop[(2 * idx + 1)*length + i] = pop[ind1*length + i];
			}
		}
	}
	else {
		// Copy individuals
		for (i = threadIdx.x;i<length;i = i + MAX_THREADS_PER_BLOCK) {
			//copy bit from parent ind1 to child 2*idx
			npop[(2 * idx)*length + i] = pop[ind1*length + i];
			//copy bit from parent ind2 to child 2*idx + 1
			npop[(2 * idx + 1)*length + i] = pop[ind2*length + i];
		}
	}

}

// Kernel invocation: tournament <<<N_BLOCK,BLOCK_LENGTH>>> (fitgpu, randomgpu,winnergpu); 
// N_BLOCK: number of blocks, such that N_BLOCK * BLOCK_LENGTH = POP_SIZE.
// BLOCK_LENGTH: length of each block, such that N_BLOCK * BLOCK_LENGTH = POP_SIZE.
// fitgpu: pointer to global memory that stores the fitness values.
// randomgpu: pointer to global memory that stores the random numbers for the tournament (2*POP_SIZE).
// winnergpu: pointer to global memory that stores the positions of the winners of the tournaments.

__global__ void tournament(float * fit, int * random, int * win) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int nro1 = random[2 * idx];
	int nro2 = random[2 * idx + 1];
	int pos;

	if (fit[nro1] > fit[nro2]) {
		pos = nro1;
	}
	else {
		pos = nro2;
	}

	win[idx] = pos;
}

// inicializa la memoria con numeros aleatorios 8 booleanos contiguos a la vez 
// de esta manera se podria asegurar que la inicializacion es la misma  para el caso no bitwise que bitwise sin tener que usar atomics en el caso bitwise
// pues el tamaño en bits del no bitwise siempre tiene que ser multiplo de 8
__global__ void initPop_device8(bool *pop, unsigned int length, unsigned long long seed) {
	unsigned int thIdx = blockIdx.x * blockDim.x + threadIdx.x;
	curandStatePhilox4_32_10_t rndState;
	curand_init(seed + thIdx, 0ull, 0ull, &rndState);
	for (unsigned int i = 8 * threadIdx.x; i < length; i = i + INIT_THREADS * 8) {
		for (int j = 0; j < 8 & (i + j < length); j++) {
			unsigned int pos = blockIdx.x * length + i + j;
			float rnd = curand_uniform(&rndState);
			pop[pos] = (rnd <= 0.5);
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
	//initPop_device32 << < POP_SIZE, INIT_THREADS >> >(*pop, len, seed);
	initPop_device8 << < POP_SIZE, INIT_THREADS >> >(*pop, len, seed);
	status.cuda = cudaGetLastError();
	if (status.failed()) return status;

	status.cuda = cudaDeviceSynchronize();
	return status;
}

