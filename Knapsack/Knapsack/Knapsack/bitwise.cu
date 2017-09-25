#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "curand_kernel.h"
#include "bitwise.cuh"

#define UNROLL
#define KNAPSACK
#define SIMULTANEO 1
#define UNBIT 1 

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


// Kernel invocation: fitness <<<POP_SIZE,MAX_THREADS_PER_BLOCK>>> (pgpu,fitgpu,REAL_LEN,MASK,CHROM_LEN);
// POP_SIZE: number of individuals of the population.
// MAX_THREADS_PER_BLOCK: constant with the maximum number of threads per block. 
// CHROM_LEN: length of the chromosome.
// pgpu: pointer to the population. 
// fitgpu: pointer to an auxiliary structure to store the fitness.
// MASK: A bit mask of the first bit. MASK = (Data)pow(2.0, (int)DataSize-1)
// REAL_LEN: real length of the chromosome of datatype Data. 
// REAL_LEN = (CHROM_LEN%DataSize)==0?CHROM_LEN/DataSize:(CHROM_LEN/DataSize + 1) 
#if UNBIT


__global__ void fitness_knapsack_b(Data * pop, float* fit, int length, int realLength, Data mask, float* W, float* G, float MAX_W, float P) {
	__shared__ float partial_g[MAX_THREADS_PER_BLOCK];  // array to store partial gain
	__shared__ float partial_w[MAX_THREADS_PER_BLOCK];  // array to store partial weights
	int idx = blockIdx.x;  // the number of the individual is indicated by the block number
	const unsigned int tid = threadIdx.x;
	partial_g[tid] = 0; // each array position is initialized in 0
	partial_w[tid] = 0; // each array position is initialized in 0

	int i, j;
	// each thread adds in partial its corresponding values
	//Data aux, shift, value;
	Data aux;
	for (i = tid;i < length;i = i + MAX_THREADS_PER_BLOCK) {
		int k = i / DataSize;
		j = (DataSize - 1) - (i - k * DataSize);//i % DataSize;
		aux = pop[idx*realLength + k];
		unsigned int value = (aux >> j) & 1;
		//shift = shift >> 1;
		partial_w[tid] = partial_w[tid] + value * W[i];
#if SIMULTANEO
		partial_g[tid] = partial_g[tid] + value * G[i];
#endif
	}
#if SIMULTANEO
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
#else
	// reduction algorithm to add the partial fitness values
	__syncthreads();

	if (MAX_THREADS_PER_BLOCK >= 1024) {
		if (tid < 512) {
			partial_w[tid] = partial_w[tid] + partial_w[tid + 512];
		}
		__syncthreads();
	}
	if (MAX_THREADS_PER_BLOCK >= 512) {
		if (tid < 256) {
			partial_w[tid] = partial_w[tid] + partial_w[tid + 256];
		}
		__syncthreads();
	}
	if (MAX_THREADS_PER_BLOCK >= 256) {
		if (tid < 128) {
			partial_w[tid] = partial_w[tid] + partial_w[tid + 128];
		}
		__syncthreads();
	}
	if (MAX_THREADS_PER_BLOCK >= 128) {
		if (tid < 64) {
			partial_w[tid] = partial_w[tid] + partial_w[tid + 64];
		}
		__syncthreads();
	}

	if (tid < 32) {
		warpReduce2(partial_w, tid);
	}

	// calcular valores de ganancia

	for (i = tid;i < length;i = i + MAX_THREADS_PER_BLOCK) {
		j = tid % DataSize;
		aux = pop[idx*realLength + i / DataSize];
		value = (aux >> j) & 1;
		shift = shift >> 1;
		partial_g[tid] = partial_g[tid] + value * G[i];
	}

	// reduction algorithm to add the partial fitness values
	__syncthreads();

	if (MAX_THREADS_PER_BLOCK >= 1024) {
		if (tid < 512) {
			partial_g[tid] = partial_g[tid] + partial_g[tid + 512];
		}
		__syncthreads();
	}
	if (MAX_THREADS_PER_BLOCK >= 512) {
		if (tid < 256) {
			partial_g[tid] = partial_g[tid] + partial_g[tid + 256];
		}
		__syncthreads();
	}
	if (MAX_THREADS_PER_BLOCK >= 256) {
		if (tid < 128) {
			partial_g[tid] = partial_g[tid] + partial_g[tid + 128];
		}
		__syncthreads();
	}
	if (MAX_THREADS_PER_BLOCK >= 128) {
		if (tid < 64) {
			partial_g[tid] = partial_g[tid] + partial_g[tid + 64];
		}
		__syncthreads();
	}




	if (tid < 32) {
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
#endif


#endif // UNBIT





// Kernel invocation: fitness <<<POP_SIZE,MAX_THREADS_PER_BLOCK>>> (pgpu,fitgpu,REAL_LEN,MASK,CHROM_LEN);
// POP_SIZE: number of individuals of the population.
// MAX_THREADS_PER_BLOCK: constant with the maximum number of threads per block. 
// CHROM_LEN: length of the chromosome.
// pgpu: pointer to the population. 
// fitgpu: pointer to an auxiliary structure to store the fitness.
// MASK: A bit mask of the first bit. MASK = (Data)pow(2.0, (int)DataSize-1)
// REAL_LEN: real length of the chromosome of datatype Data. 
// REAL_LEN = (CHROM_LEN%DataSize)==0?CHROM_LEN/DataSize:(CHROM_LEN/DataSize + 1) 

#if UNBIT == false // !UNBIT
__global__ void fitness_knapsack_b(Data * pop, float* fit, int length, int realLength, Data mask, float* W, float* G, float MAX_W, float P) {
	__shared__ float partial_g[MAX_THREADS_PER_BLOCK];  // array to store partial gain
	__shared__ float partial_w[MAX_THREADS_PER_BLOCK];  // array to store partial weights
	int idx = blockIdx.x;  // the number of the individual is indicated by the block number
	const unsigned int tid = threadIdx.x;
	partial_g[tid] = 0; // each array position is initialized in 0
	partial_w[tid] = 0; // each array position is initialized in 0

	int i, j;
	// each thread adds in partial its corresponding values
	Data aux, shift, value;
	for (i = tid;i < realLength;i = i + MAX_THREADS_PER_BLOCK) {
		aux = pop[idx*realLength + i];
		shift = mask;
		for (j = 0;j < DataSize && (i*DataSize + j < length);j = j + 1) {
			value = (aux & shift) == 0 ? 0 : 1;
			shift = shift >> 1;
			partial_w[tid] = partial_w[tid] + value * W[i*DataSize + j];
#if SIMULTANEO
			partial_g[tid] = partial_g[tid] + value * G[i*DataSize + j];
#endif
		}
	}
#if SIMULTANEO
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
#else
	// reduction algorithm to add the partial fitness values
	__syncthreads();

	if (MAX_THREADS_PER_BLOCK >= 1024) {
		if (tid < 512) {
			partial_w[tid] = partial_w[tid] + partial_w[tid + 512];
		}
		__syncthreads();
	}
	if (MAX_THREADS_PER_BLOCK >= 512) {
		if (tid < 256) {
			partial_w[tid] = partial_w[tid] + partial_w[tid + 256];
		}
		__syncthreads();
	}
	if (MAX_THREADS_PER_BLOCK >= 256) {
		if (tid < 128) {
			partial_w[tid] = partial_w[tid] + partial_w[tid + 128];
		}
		__syncthreads();
	}
	if (MAX_THREADS_PER_BLOCK >= 128) {
		if (tid < 64) {
			partial_w[tid] = partial_w[tid] + partial_w[tid + 64];
		}
		__syncthreads();
	}

	if (tid < 32) {
		warpReduce2(partial_w, tid);
	}

	// calcular valores de ganancia

	for (i = tid;i < realLength;i = i + MAX_THREADS_PER_BLOCK) {
		aux = pop[idx*realLength + i];
		shift = mask;
		for (j = 0;j < DataSize && (i*DataSize + j < length);j = j + 1) {
			value = (aux & shift) == 0 ? 0 : 1;
			shift = shift >> 1;
			partial_g[tid] = partial_g[tid] + value * G[i*DataSize + j];
		}
	}



	// reduction algorithm to add the partial fitness values
	__syncthreads();

	if (MAX_THREADS_PER_BLOCK >= 1024) {
		if (tid < 512) {
			partial_g[tid] = partial_g[tid] + partial_g[tid + 512];
		}
		__syncthreads();
	}
	if (MAX_THREADS_PER_BLOCK >= 512) {
		if (tid < 256) {
			partial_g[tid] = partial_g[tid] + partial_g[tid + 256];
		}
		__syncthreads();
	}
	if (MAX_THREADS_PER_BLOCK >= 256) {
		if (tid < 128) {
			partial_g[tid] = partial_g[tid] + partial_g[tid + 128];
		}
		__syncthreads();
	}
	if (MAX_THREADS_PER_BLOCK >= 128) {
		if (tid < 64) {
			partial_g[tid] = partial_g[tid] + partial_g[tid + 64];
		}
		__syncthreads();
	}




	if (tid < 32) {
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
#endif 
#endif // !UNBIT






// Kernel invocation: spx <<POP_SIZE/2,MAX_THREADS_PER_BLOCK>>> (oldpop,newpop,positions,randomPC,randomPoint,
// REAL_LEN,PROB_CROSS); 
// POP_SIZE: number of individuals of the population.
// MAX_THREADS_PER_BLOCK: constant with the maximum number of threads per block. 
// oldpop: pointer to global memory that stores the old population. 
// newpop: pointer to global memory that stores the new population.
// positions: pointer to global memory that stores the indexes of individuals for crossover
// randomPC: pointer to global memory that stores the random values for crossover.
// randomPoint: pointer to global memory that stores the random points for crossover.
// REAL_LEN: real length of the chromosome of datatype Data. 
// REAL_LEN = (CHROM_LEN%DataSize)==0?CHROM_LEN/DataSize:(CHROM_LEN/DataSize + 1) 
// PROB_CROSS: crossover probability. 

__global__ void spx_b(Data *pop, Data *npop, int *pos, float *randomPC, int *randomPoint, int length, float PROB_CROSS){

  int idx = blockIdx.x; // the number of the individual is indicated by the block number
  int ind1 = pos[2*idx]; // index of first individual for crossover
  int ind2 = pos[2*idx+1]; // index of second individual for crossover
  float pc = randomPC[idx]; // value for crossover
  int pnt = randomPoint[idx]; // crossover point
  int i;
	
  if (pc <= PROB_CROSS) {
     // Cross individuals
     for(i=threadIdx.x;i<length;i=i+MAX_THREADS_PER_BLOCK) {
	  if (i<(pnt/DataSize)) {	  
              //copy word from parent ind1 to child 2*idx
              npop[2*idx*length+i] = pop[ind1*length+i];
              //copy word from parent ind2 to child 2*idx + 1
              npop[(2*idx+1)*length+i] = pop[ind2*length+i];
	  } else if (i==(pnt/DataSize) && pnt % DataSize > 0) {
	         unsigned int word = pnt/DataSize;
	         unsigned int move = pnt % DataSize;
	         unsigned int moveComp = DataSize - move;
                 npop[2*idx*length+word] = ((Data)(pop[ind1*length+word] >> moveComp) << moveComp) | ((Data)(pop[ind2*length+word] << move) >>  move);
                 npop[(2*idx+1)*length+word] = ((Data)(pop[ind2*length+word] >> moveComp) << moveComp) | ((Data)(pop[ind1*length+word] << move) >>  move);	  
	  } else {
	    //copy word from parent ind2 to child 2*idx
            npop[2*idx*length+i] = pop[ind2*length+i];
	    //copy word from parent ind1 to child 2*idx + 1
            npop[(2*idx+1)*length+i] = pop[ind1*length+i];
	  }
     }
 } else {
      // Copy individuals
      for(i=threadIdx.x;i<length;i=i+MAX_THREADS_PER_BLOCK) {
         //copy word from parent ind1 to child 2*idx
	 npop[(2*idx)*length+i] = pop[ind1*length+i];
	 //copy word from parent ind2 to child 2*idx + 1
	 npop[(2*idx+1)*length+i] = pop[ind2*length+i];
      }
 }

}



// Kernel invocation: dpx <<POP_SIZE/2,MAX_THREADS_PER_BLOCK>>> (oldpop,newpop,positions,randomPC,randomPoint,
// REAL_LEN,PROB_CROSS); 
// POP_SIZE: number of individuals of the population.
// MAX_THREADS_PER_BLOCK: constant with the maximum number of threads per block. 
// oldpop: pointer to global memory that stores the old population. 
// newpop: pointer to global memory that stores the new population.
// positions: pointer to global memory that stores the indexes of individuals for crossover
// randomPC: pointer to global memory that stores the random values for crossover.
// randomPoint: pointer to global memory that stores the random points for crossover.
// REAL_LEN: real length of the chromosome of datatype Data. 
// REAL_LEN = (CHROM_LEN%DataSize)==0?CHROM_LEN/DataSize:(CHROM_LEN/DataSize + 1) 
// PROB_CROSS: crossover probability. 

__global__ void dpx_b(Data *pop, Data *npop, int *pos, float *randomPC, int *randomPoint, int length, float PROB_CROSS) {

	int idx = blockIdx.x;  // the number of the individual is indicated by the block number
	int ind1 = pos[2 * idx];  // index of first individual for crossover
	int ind2 = pos[2 * idx + 1]; // index of second individual for crossover
	float pc = randomPC[idx]; // value for crossover
	int pnt1 = randomPoint[2 * idx]; // first crossover point
	int pnt2 = randomPoint[2 * idx + 1]; // second crossover point. pnt2 != pnt1

#if ORDEN_DPX
	pnt1 = min(pnt1, pnt2);
	pnt2 = max(pnt1, pnt2);
#endif // ORDEN_DPX
	if (pnt1 == pnt2) {
		if (pnt1 > 0) pnt1--;
		else pnt2++;
	}

	int word1 = pnt1 / DataSize;     // word of the first crossover point
	unsigned int wordPos1 = pnt1 % DataSize; // position in the word of the first crossover point 
	int word2 = pnt2 / DataSize;     // word of the second crossover point
	unsigned int wordPos2 = pnt2 % DataSize; // position in the word of the second crossover point 


	int i;

	if (pc <= PROB_CROSS) {
		// Cross individuals
		if (word1 != word2) {
			// los puntos de cruce estan en palabras distintas
			unsigned int move;
			for (i = threadIdx.x;i < length;i = i + MAX_THREADS_PER_BLOCK) {
				if (i<word1 || i>word2 || (i == word2 && wordPos2 == 0)) {
					//copy word from parent ind1 to child 2*idx
					npop[2 * idx*length + i] = pop[ind1*length + i];
					//copy word from parent ind2 to child 2*idx + 1
					npop[(2 * idx + 1)*length + i] = pop[ind2*length + i];
				}
				else if (i == word1 && wordPos1 > 0) { // primer punto de cruce, combino las palabras de ambos individuos
					 // the word has to be shifted  
					move = DataSize - wordPos1;
					// 1er hijo: Los wordsPos1 bits superiores  son los del 1er individuo, mientras que los (DataSize - wordPos1) inferiores son del 2do individuo
					// 1er hijo: | 1er padre | 2do padre |
					npop[2 * idx*length + i] = ((Data)(pop[ind1*length + i] >> move) << move) | ((Data)(pop[ind2*length + i] << wordPos1) >> wordPos1);
					// 2do ijo: Los wordsPos1 bits superiores  son los del 2do individuo, mientras que los (DataSize - wordPos1) inferiores son del 1er individuo
					// 2do hijo: | 2do padre | 1er padre |
					npop[(2 * idx + 1)*length + i] = ((Data)(pop[ind2*length + i] >> move) << move) | ((Data)(pop[ind1*length + i] << wordPos1) >> wordPos1);
				}
				else if (i == word2 && wordPos2 > 0) { // segundo punto de cruce
			 // the word has to be shifted
					move = DataSize - wordPos2;
					// 1er hijo: Los wordPos2 bits superiores son del 2do individuo, mientras que los (DataSize - wordPos2) inferiores son del 1er individuo
					npop[2 * idx*length + i] = ((Data)(pop[ind2*length + i] >> move) << move) | ((Data)(pop[ind1*length + i] << wordPos2) >> wordPos2);
					// 2do hijo: Los wordPos2 bits superiores son del 1er individuo, mientras que los (DataSize - wordPos2) inferiores son del 2do individuo
					npop[(2 * idx + 1)*length + i] = ((Data)(pop[ind1*length + i] >> move) << move) | ((Data)(pop[ind2*length + i] << wordPos2) >> wordPos2);
				}
				else {
					//copy word from parent ind2 to child 2*idx	       		    
					npop[2 * idx*length + i] = pop[ind2*length + i];
					//copy word from parent ind1 to child 2*idx + 1
					npop[(2 * idx + 1)*length + i] = pop[ind1*length + i];
				}
			}
		}
		else { // wordPos1 != wordPos2 since pnt2 != pnt1
			// ambos puntos de cruce estan en la misma palabra
			for (i = threadIdx.x;i < length;i = i + MAX_THREADS_PER_BLOCK) {
				if (i < word1) {
					//copy word from parent ind1 to child 2*idx
					npop[2 * idx*length + i] = pop[ind1*length + i];
					//copy word from parent ind2 to child 2*idx + 1
					npop[(2 * idx + 1)*length + i] = pop[ind2*length + i];
				}
				else if (i == word1) {
					// the word has to be shifted 
					unsigned int move = DataSize - wordPos1;
					unsigned int move2 = DataSize - wordPos2;
					// rest1 = (DataSize - wordPos1) bits inferiores del individuo 2
					Data rest1 = ((Data)(pop[ind2*length + i] << wordPos1) >> wordPos1);
					// rest2 = (DataSize - wordPos1) bits inferiores del individuo 1
					Data rest2 = ((Data)(pop[ind1*length + i] << wordPos1) >> wordPos1);
					if (wordPos1 != 0) {
						// hijo1: | padre1 (wp1) bits sup | padre 2 (DS - wp1 - wp2) del medio | padre 1 (wp2) bits inferiores |
						npop[2 * idx*length + i] = ((Data)(pop[ind1*length + i] >> move) << move) | 
													((Data)(rest1 >> move2) << move2) |
													((Data)(rest2 << wordPos2) >> wordPos2);

						// hijo2: | padre2 (wp1) bits sup | padre 2 (DS - wp1 - wp2) del medio | padre 1 (DS - wp1 - wp2) bits inferiores |
						npop[(2 * idx + 1)*length + i] = ((Data)(pop[ind2*length + i] >> move) << move) |
														((Data)(rest2 >> move2) << move2) |
														((Data)(rest1 << wordPos2) >> wordPos2);
					}
					else {
						npop[2 * idx*length + i] = ((Data)(rest1 >> move2) << move2) | ((Data)(rest2 << wordPos2) >> wordPos2);
						npop[(2 * idx + 1)*length + i] = ((Data)(rest2 >> move2) << move2) | ((Data)(rest1 << wordPos2) >> wordPos2);
					}
				}
				else {
					//copy word from parent ind2 to child 2*idx	       		    
					npop[2 * idx*length + i] = pop[ind2*length + i];
					//copy word from parent ind1 to child 2*idx + 1
					npop[(2 * idx + 1)*length + i] = pop[ind1*length + i];
				}
			}
		}
	}
	else {
		// Copy individuals
		for (i = threadIdx.x;i < length;i = i + MAX_THREADS_PER_BLOCK) {
			//copy word from parent ind1 to child 2*idx
			npop[(2 * idx)*length + i] = pop[ind1*length + i];
			//copy word from parent ind2 to child 2*idx + 1
			npop[(2 * idx + 1)*length + i] = pop[ind2*length + i];
		}
	}
}


// Kernel invocation: mutation <<<N_BLOCK,BLOCK_LENGTH>>> (pgpu,randomPM,randomPoint,REAL_LEN,MASK,PROB_MUT);
// N_BLOCK: number of blocks, such that N_BLOCK * BLOCK_LENGTH = POP_SIZE.
// BLOCK_LENGTH: length of each block, such that N_BLOCK * BLOCK_LENGTH = POP_SIZE.
// pgpu: pointer to global memory that stores the population. 
// randomPM: pointer to global memory that stores the random values for mutation.
// randomPoint: pointer to global memory that stores the random points for mutation.
// REAL_LEN: real length of the chromosome of datatype Data. 
// REAL_LEN = (CHROM_LEN%DataSize)==0?CHROM_LEN/DataSize:(CHROM_LEN/DataSize + 1) 
// MASK: A bit mask of the first bit. MASK = (Data)pow(2.0, (int)DataSize-1)
// PROB_MUT: mutation probability. 

__global__ void mutation_b(Data *pop, float *randomPM, int *randomPoint, int length, Data mask, float PROB_MUT){
	
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float pm = randomPM[idx];   // value for mutation
    int pnt = randomPoint[idx]; // mutation point
	
    if (pm <= PROB_MUT) {
          int word = pnt/DataSize;
	  int wordPos = pnt % DataSize;
          Data aux1 = pop[idx*length+word];
	  Data aux2 = mask >> wordPos;
	  pop[idx*length+word] = aux1 ^ aux2;
    }
 
}




// Kernel invocation: tournament <<<N_BLOCK,BLOCK_LENGTH>>> (fitgpu, randomgpu,winnergpu); 
// N_BLOCK: number of blocks, such that N_BLOCK * BLOCK_LENGTH = POP_SIZE.
// BLOCK_LENGTH: length of each block, such that N_BLOCK * BLOCK_LENGTH = POP_SIZE.
// fitgpu: pointer to global memory that stores the fitness values.
// randomgpu: pointer to global memory that stores the random numbers for the tournament (2*POP_SIZE).
// winnergpu: pointer to global memory that stores the positions of the winners of the tournaments.

__global__ void tournament_b(float * fit, int * random, int * win) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nro1 = random[2*idx];
    int nro2 = random[2*idx+1];
    int pos;
    
    if (fit[nro1] > fit[nro2]) {
	pos = nro1;
    } else {
	pos = nro2;
    }

    win[idx] = pos;
}


/*__global__ void initPop_device(bool *pop, unsigned int length, unsigned long long seed) {
	unsigned int thIdx = blockIdx.x * blockDim.x + threadIdx.x;
	curandStatePhilox4_32_10_t rndState;
	curand_init(seed + thIdx, 0ull, 0ull, &rndState);
	for (unsigned int i = threadIdx.x; i < length; i = i + INIT_THREADS) {
		unsigned int pos = blockIdx.x * length + i;
		pop[pos] = (curand_uniform(&rndState) <= 0.5);
	}
}*/


/*__global__ void initPop_device32_bitwise(Data *pop,  unsigned int dataLength, int length,unsigned long long seed) {
	unsigned int thIdx = blockIdx.x * blockDim.x + threadIdx.x;
	curandStatePhilox4_32_10_t rndState;
	curand_init(seed + thIdx, 0ull, 0ull, &rndState);
	unsigned int i;
	for (i = threadIdx.x; i < dataLength; i++ ) {
#if DataSize <= 32

			uint32_t rnd = curand(&rndState);
			for (int j = 0; j < DataSize / 8; j++) {

			}
			pop[blockIdx.x * dataLength + i] = (Data)(rnd & DataMask);
#else
			uint32_t rnd1 = curand(&rndState);
			uint32_t rnd2 = curand(&rndState);
			pop[blockIdx.x * dataLength + i] = ((((uint64_t)rnd1) << 32) + rnd2);
#endif
			if (i == dataLength - 1) {
				// poner en 0 bits sobrantes (cuales???)
			}
	}
	

}*/

#define datasize_bytes (DataSize / 8)
#define posMask (datasize_bytes - 1)

//INICIALIZACIÓN ADAPTATIVA
// inicializa la memoria con numeros aleatorios 8 booleanos contiguos a la vez 
// de esta manera se podria asegurar que la inicializacion es la misma  para el caso no bitwise que bitwise sin tener que usar atomics en el caso bitwise
// pues el tamaño en bits del no bitwise siempre tiene que ser multiplo de 8
__global__ void initPop_device8_bitwise(Data *pop, unsigned int byteLength, int length, unsigned long long seed) {
	unsigned int thIdx = blockIdx.x * blockDim.x + threadIdx.x;
	curandStatePhilox4_32_10_t rndState;
	curand_init(seed + thIdx, 0ull, 0ull, &rndState);
	unsigned char *bytes = reinterpret_cast<unsigned char*>(pop);
	// itera sobre los bytes de cada individuo de forma intercalada , avanzando cada thread de a INIT_THREADS bytes
	for (unsigned int i = threadIdx.x; i < byteLength; i = i + INIT_THREADS) {

		// setea la posicion tomando en cuanta que cuda (al igual que la mayoria de los cpu) usa little endian
		// es decir si usa uint32_t los bytes quedan en el orden 3 2 1 0, no 0 1 2 3
		unsigned int lastPos = i & posMask;
		//unsigned int posEndian = (i / datasize_bytes) * datasize_bytes + (datasize_bytes - 1) - i % datasize_bytes;
		unsigned int posEndian = (i - lastPos) + (posMask - lastPos);
		unsigned int pos = blockIdx.x * byteLength + posEndian;
		bytes[pos] = 0;
		// por cada byte setear los bits
		for (int j = 0; j < 8 & (i * 8 + j < length); j++) {
			int shift = 7 - j;
			float rnd = curand_uniform(&rndState);
			unsigned char bit = (rnd <= 0.5) ? 1 : 0;
			bytes[pos] += bit << shift;
		}
	}
}


ErrorInfo generatePOP_device_bitwise(unsigned long seed, size_t POP_SIZE, int len, Data** pop, Data** npop) {
	ErrorInfo status;
	uint64_t dataLen = (len + DataSize - 1) / DataSize;
	uint64_t byteLen = dataLen * DataSize / 8;
	size_t N = POP_SIZE * dataLen;
	status.cuda = cudaMalloc(pop, N * sizeof(Data));
	if (status.failed()) return status;
	status.cuda = cudaMalloc(npop, N * sizeof(Data));
	if (status.failed()) return status;
	
	initPop_device8_bitwise << < POP_SIZE, INIT_THREADS >> >(*pop, byteLen, len, seed);
	status.cuda = cudaGetLastError();
	if (status.failed()) return status;

	status.cuda = cudaDeviceSynchronize();
	return status;
}