#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "notbitwise.cuh"

// Kernel invocation: mutation <<<N_BLOCK,BLOCK_LENGTH>>> (pgpu,randomPM,randomPoint,CHROM_LEN,PROB_MUT);
// N_BLOCK: number of blocks, such that N_BLOCK * BLOCK_LENGTH = POP_SIZE.
// BLOCK_LENGTH: length of each block, such that N_BLOCK * BLOCK_LENGTH = POP_SIZE.
// pgpu: pointer to global memory that stores the population. 
// randomPM: pointer to global memory that stores the random values for mutation.
// randomPoint: pointer to global memory that stores the random points for mutation.
// CHROM_LEN: length of the chromosome.
// PROB_MUT: mutation probability. 

__global__ void mutation(bool *pop, float *randomPM, int *randomPoint, int totalLength, float PROB_MUT) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float pm = randomPM[idx];   // value for mutation
	int pnt = randomPoint[idx]; // mutation point

	if (pm <= PROB_MUT) {
		pop[idx*totalLength + pnt] = !pop[idx*totalLength + pnt];
	}

}


// Kernel invocation: fitness <<<POP_SIZE,MAX_THREADS_PER_BLOCK>>> (pgpu,fitgpu,CHROM_LEN);
// POP_SIZE: number of individuals of the population.
// MAX_THREADS_PER_BLOCK: constant with the maximum number of threads per block. 
// CHROM_LEN: length of the chromosome.
// pgpu: pointer to global memory that stores the population. 
// fitgpu: pointer to global memory that stores the fitness values.

__global__ void fitness(bool * pop, int * fit, int length, int totalLength) {

	__shared__ int partial[MAX_THREADS_PER_BLOCK];  // array to store partial fitness
	int idx = blockIdx.x;  // the number of the individual is indicated by the block number
	partial[threadIdx.x] = 0; // each array position is initialized in 0

	int i;
	// each thread adds in partial its corresponding values
	for (i = threadIdx.x;i<length;i = i + MAX_THREADS_PER_BLOCK) {
		partial[threadIdx.x] = partial[threadIdx.x] + pop[idx*totalLength + i];
	}

	// reduction algorithm to add the partial fitness values
	__syncthreads();
	i = MAX_THREADS_PER_BLOCK / 2;
	while (i != 0) {
		if (threadIdx.x < i)
			partial[threadIdx.x] = partial[threadIdx.x] + partial[threadIdx.x + i];
		__syncthreads();
		i = i / 2;
	}

	// finally thread 0 writes the fitness value in global memory
	if (threadIdx.x == 0)
		fit[idx] = partial[0];
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


#define VectorType int4
#define VectorSize 16
#define VectorMask (VectorSize - 1)
#define InverseMask (~VectorMask)

__device__ bool aligned(int idx) {
	return idx & VectorMask == 0;
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
// vecLength = ceil( CHROM_LEN / VectorSize),  CHROM_LEN: length of the chromosome.
// PROB_CROSS: crossover probability. 

__global__ void spx_vectorized(bool *pop, bool *npop, int *pos, float *randomPC, int *randomPoint, int vecLength, float PROB_CROSS) {

	int idx = blockIdx.x; // the number of the individual is indicated by the block number
	int ind1 = pos[2 * idx]; // index of first individual for crossover
	int ind2 = pos[2 * idx + 1]; // index of second individual for crossover
	float pc = randomPC[idx]; // value for crossover
	int pnt = randomPoint[idx]; // crossover point
	int i;
	int pnt_vec = pnt / VectorSize;
	if (pc <= PROB_CROSS) {
		// Cross individuals
		for (i = threadIdx.x;i < vecLength ;i = i + MAX_THREADS_PER_BLOCK) {

			if (i< pnt_vec) {
				//copy bit from parent ind1 to child 2*idx
				reinterpret_cast<VectorType*>(npop)[2 * idx*vecLength + i] = reinterpret_cast<VectorType*>(pop)[ind1*vecLength + i];
				//copy bit from parent ind2 to child 2*idx + 1
				reinterpret_cast<VectorType*>(npop)[(2 * idx + 1)*vecLength + i] = reinterpret_cast<VectorType*>(pop)[ind2*vecLength + i];
			}
			else if (i > pnt_vec) {
				//copy bit from parent ind2 to child 2*idx
				reinterpret_cast<VectorType*>(npop)[2 * idx*vecLength + i] = reinterpret_cast<VectorType*>(pop)[ind2*vecLength + i];
				//copy bit from parent ind1 to child 2*idx + 1
				reinterpret_cast<VectorType*>(npop)[(2 * idx + 1)*vecLength + i] = reinterpret_cast<VectorType*>(pop)[ind1*vecLength + i];
			}
			else {
				// non vectorized copy
				int totalLength = vecLength * VectorSize;
				for (int j = i * VectorSize; j < pnt; j++) {
					//copy bit from parent ind1 to child 2*idx
					npop[(2 * idx) * totalLength  + j] = pop[ind1*totalLength + j];
					//copy bit from parent ind2 to child 2*idx + 1
					npop[(2 * idx + 1)*totalLength + j] = pop[ind2*totalLength + j];
				}
				for (int j = pnt; j < (i + 1) * VectorSize; j++) {
					//copy bit from parent ind2 to child 2*idx
					npop[(2 * idx)*totalLength + j] = pop[ind2*totalLength + j];
					//copy bit from parent ind1 to child 2*idx + 1
					npop[(2 * idx + 1)*totalLength + j] = pop[ind1*totalLength + j];
				}
			}
		}
	}
	else {
		// Copy individuals (fully vectorized)
		for (i = threadIdx.x;i<vecLength;i = i + MAX_THREADS_PER_BLOCK) {
			//copy bit from parent ind1 to child 2*idx
			reinterpret_cast<VectorType*>(npop)[(2 * idx)*vecLength + i] = reinterpret_cast<VectorType*>(pop)[ind1*vecLength + i];
			//copy bit from parent ind2 to child 2*idx + 1
			reinterpret_cast<VectorType*>(npop)[(2 * idx + 1)*vecLength + i] = reinterpret_cast<VectorType*>(pop)[ind2*vecLength + i];
		}
	}

}


__global__ void dpx_vectorized(bool *pop, bool *npop, int *pos, float *randomPC, int *randomPoint, int vecLength, float PROB_CROSS) {

	int idx = blockIdx.x;  // the number of the individual is indicated by the block number
	int ind1 = pos[2 * idx];  // index of first individual for crossover
	int ind2 = pos[2 * idx + 1]; // index of second individual for crossover
	float pc = randomPC[idx]; // value for crossover
	int pnt1 = randomPoint[2 * idx]; // first crossover point
	int pnt2 = randomPoint[2 * idx + 1]; // second crossover point
	int i;

	int pnt_vec1 = pnt1 / VectorSize;
	int pnt_vec2 = pnt2 / VectorSize;

	if (pc <= PROB_CROSS) {
		// Cross individuals
		for (i = threadIdx.x;i < vecLength;i = i + MAX_THREADS_PER_BLOCK) {

			if (i< pnt_vec1 || i > pnt_vec2) {
				//copy bit from parent ind1 to child 2*idx
				reinterpret_cast<VectorType*>(npop)[2 * idx*vecLength + i] = reinterpret_cast<VectorType*>(pop)[ind1*vecLength + i];
				//copy bit from parent ind2 to child 2*idx + 1
				reinterpret_cast<VectorType*>(npop)[(2 * idx + 1)*vecLength + i] = reinterpret_cast<VectorType*>(pop)[ind2*vecLength + i];
			}
			else if (i > pnt_vec1 && i < pnt_vec2) {
				//copy bit from parent ind2 to child 2*idx
				reinterpret_cast<VectorType*>(npop)[2 * idx*vecLength + i] = reinterpret_cast<VectorType*>(pop)[ind2*vecLength + i];
				//copy bit from parent ind1 to child 2*idx + 1
				reinterpret_cast<VectorType*>(npop)[(2 * idx + 1)*vecLength + i] = reinterpret_cast<VectorType*>(pop)[ind1*vecLength + i];
			}
			else {
				// non vectorized copy
				int totalLength = vecLength * VectorSize;
				for (int j = i * VectorSize; j < i * VectorSize + VectorSize; j++) {
					if (j<pnt1 || j >= pnt2) {
						//copy bit from parent ind1 to child 2*idx
						npop[2 * idx*totalLength + j] = pop[ind1*totalLength + j];
						//copy bit from parent ind2 to child 2*idx + 1
						npop[(2 * idx + 1)*totalLength + j] = pop[ind2*totalLength + j];
					}
					else {
						//copy bit from parent ind2 to child 2*idx	       		    
						npop[2 * idx*totalLength + j] = pop[ind2*totalLength + j];
						//copy bit from parent ind1 to child 2*idx + 1
						npop[(2 * idx + 1)*totalLength + j] = pop[ind1*totalLength + j];
					}
				}
			}
		}
	}
	else {
		// Copy individuals (fully vectorized)
		for (i = threadIdx.x;i<vecLength;i = i + MAX_THREADS_PER_BLOCK) {
			//copy bit from parent ind1 to child 2*idx
			reinterpret_cast<VectorType*>(npop)[(2 * idx)*vecLength + i] = reinterpret_cast<VectorType*>(pop)[ind1*vecLength + i];
			//copy bit from parent ind2 to child 2*idx + 1
			reinterpret_cast<VectorType*>(npop)[(2 * idx + 1)*vecLength + i] = reinterpret_cast<VectorType*>(pop)[ind2*vecLength + i];
		}
	}

}