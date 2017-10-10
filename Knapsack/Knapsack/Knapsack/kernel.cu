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
#include "evaluacion.cuh"

#ifdef __INTELLISENSE__

//for __syncthreads()
#ifndef __CUDACC_RTC__ 
#define __CUDACC_RTC__
#endif // !(__CUDACC_RTC__)
//for atomicAdd
#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__

#define __DEVICE_FUNCTIONS_H__

#endif


#define makeRandomInts makeRandomIntegers2

struct EvalInfo {
	float min;
	float minValido;
	float max;
	float avg;
	float avgValido;
	float avgPenal;
	int invalidos;
};

float timeFitness, timeCross;

bool KERNEL_TIMING = false;
bool UNIT_TEST = false;
int SALIDA_STEP = 500; // cada cuanto mostrar estadisticas
bool MAX_DETALLADO = true; // calcular siempre maximo
bool CALC_MAXFIT = true;




__global__ void scaleRandom(float* floatRnd, int* intRnd, size_t N, unsigned int scale) {
	unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos < N) {
		intRnd[pos] = __float2int_rd(floatRnd[pos] * (scale + 0.999999f));
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


__global__ void scaleRandom2(uint32_t* rnd,  size_t N, double scale) {
	unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos < N) {
		// multiplicar aleatorio por escala, mul
		rnd[pos] = __double2uint_rd(__dmul_rd (rnd[pos] , scale));
	}

}

__global__ void scaleRandomMod(uint32_t* rnd, size_t N, uint32_t max1) {
	unsigned int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos < N) {
		rnd[pos] = rnd[pos] % max1;
	}

}



ErrorInfo makeRandomIntegers2(curandGenerator_t& generator, int32_t* indices, size_t N, unsigned int max) {
	ErrorInfo status;
	uint32_t* uindices = (uint32_t*)indices; // reinterpreto indices como si fueran unsigned
	double scale = (double)(max + (1 - 1e-6)) / ((1LL << 32) - 1) ;

	status.curand = curandGenerate(generator, uindices, N);
	status.cuda = cudaDeviceSynchronize();
	if (status.failed()) return status;

	unsigned int blocks = ((unsigned int)N + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK; // ceil(N/MAX_THREADS_PER_BLOCK)
	scaleRandom2 << <blocks, MAX_THREADS_PER_BLOCK >> >( uindices, N, scale);
	status.cuda = cudaGetLastError();
	if (status.failed()) return status;

	status.cuda = cudaDeviceSynchronize();

	return status;

}

ErrorInfo makeRandomIntegersMod(curandGenerator_t& generator, int32_t* indices, unsigned int N, unsigned int max) {
	ErrorInfo status;
	uint32_t* uindices = (uint32_t*)indices; // reinterpreto indices como si fueran unsigned

	status.curand = curandGenerate(generator, uindices, N);
	status.cuda = cudaDeviceSynchronize();
	if (status.failed()) return status;

	unsigned int blocks = (N + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK; // ceil(N/MAX_THREADS_PER_BLOCK)
	scaleRandomMod << <blocks, MAX_THREADS_PER_BLOCK >> >(uindices, N, max+1);
	status.cuda = cudaGetLastError();
	if (status.failed()) return status;

	status.cuda = cudaDeviceSynchronize();

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

	status = makeRandomInts(generator, randomPoint, POP_SIZE, len - 1);
	if (status.failed()) return status;
	status.cuda = cudaDeviceSynchronize();

	return status;

}



ErrorInfo makeRandomNumbersSpx(curandGenerator_t& generator, size_t POP_SIZE, int len, float* randomPC, int* randomPoint) {
	ErrorInfo status;
	size_t HALF_SIZE = POP_SIZE / 2;

	status = makeRandomInts(generator, randomPoint, HALF_SIZE, len - 1);
	if (status.failed()) return status;

	status.curand = curandGenerateUniform(generator, randomPC, HALF_SIZE);
	status.cuda = cudaDeviceSynchronize();

	return status;

}

ErrorInfo makeRandomNumbersDpx(curandGenerator_t& generator, size_t POP_SIZE, int len, float* randomPC, int* randomPoint) {
	ErrorInfo status;
	size_t HALF_SIZE = POP_SIZE / 2;

	status = makeRandomInts(generator, randomPoint, POP_SIZE, len - 1);
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
	return makeRandomInts(generator, random, POP_SIZE * 2, POP_SIZE - 1);
}

cudaError_t InitFit(float** dev_fit,size_t POP_SIZE) {
	return cudaMalloc(dev_fit, sizeof(float) * POP_SIZE);
}
cudaError_t InitWin(int** dev_win, size_t POP_SIZE) {
	return cudaMalloc(dev_win, sizeof(int) * POP_SIZE );
}


void printInfo(int gen, const EvalInfo& eval) {
	printf("gen %d: Inval: %d, AvgP: %.2f, MinV: %.0f, AvgV: %.1f, Max: %.0f\n", gen, eval.invalidos, eval.avgPenal, eval.minValido, eval.avgValido, eval.max);
}

float* mem1;
void initMemory() {
	cudaMallocHost(&mem1, sizeof(float));
}

// evaluate comun

ErrorInfo evaluate_(size_t POP_SIZE, float* dev_fit, EvalInfo& eval, int gen) {

	bool mostrar = (SALIDA && (gen % SALIDA_STEP) == 0);
	ErrorInfo status;
	float minFitVal; //avgFit, avgFitVal, ,cantInv;
	T_FIT maxFit;
	float host_stats[3];
	size_t STAT_SIZE = sizeof(float) * 3;

	status.cuda = cudaGetLastError();
	if (status.failed()) return status;

	float* out1;
	float* out3;

	int nroBlocks = ((unsigned int)POP_SIZE + MAX_THREADS_PER_BLOCK - 1) / (MAX_THREADS_PER_BLOCK);
	cudaMalloc(&out1, nroBlocks * sizeof(float));
	cudaMalloc(&out3, STAT_SIZE);



	// hallar maximo (por defecto se calcula siempre para conocer mejor iteracion)
	if (mostrar | MAX_DETALLADO) {
		maximo << <nroBlocks, MAX_THREADS_PER_BLOCK >> >(dev_fit, out1, (unsigned int)POP_SIZE);
		maximo << <1, MAX_THREADS_PER_BLOCK >> >(out1, out1, nroBlocks);
		// para copias suficientemente pequeñas no se observo mejor performance en usar memoria pinned
		cudaMemcpy(&maxFit, out1, sizeof(T_FIT), cudaMemcpyDeviceToHost); //cudaMemcpy(mem1, out1, sizeof(T_FIT), cudaMemcpyDeviceToHost);
	}

	eval.max = maxFit; 
	

	// hallar minimo
	/*minimo << <nroBlocks, MAX_THREADS_PER_BLOCK >> >(dev_fit, out1, POP_SIZE);
	minimo << <1, MAX_THREADS_PER_BLOCK >> >(out1, out1, nroBlocks);
	cudaMemcpy(&minFit, out1, sizeof(T_FIT), cudaMemcpyDeviceToHost);*/

	// promedio
	/*sumar << <nroBlocks, MAX_THREADS_PER_BLOCK >> >(dev_fit, out1, POP_SIZE);
	sumar << <1, MAX_THREADS_PER_BLOCK >> >(out1, out1, nroBlocks);
	cudaMemcpy(&avgFit, out2, sizeof(float), cudaMemcpyDeviceToHost);*/

	// calcular otras estadisticas solo si se muestran
	if (mostrar) {
		// cant invalidos, promedio validos e invalidos
		sumarValidez << <1, MAX_THREADS_PER_BLOCK >> >(dev_fit, out3, POP_SIZE);
		cudaMemcpy(&host_stats, out3, STAT_SIZE, cudaMemcpyDeviceToHost);
		eval.invalidos = (int)host_stats[0];
		eval.avgPenal = eval.invalidos > 0 ? host_stats[1] / eval.invalidos : 0;
		eval.avgValido = eval.invalidos < POP_SIZE ? host_stats[2] / (POP_SIZE - eval.invalidos) : 0;

		// minimo validos
		minimoValidos << <1, MAX_THREADS_PER_BLOCK >> >(dev_fit, out1, POP_SIZE);
		cudaMemcpy(&minFitVal, out1, sizeof(float), cudaMemcpyDeviceToHost);
		eval.minValido = minFitVal;
	}

	status.cuda = cudaGetLastError();
	if (status.failed()) return status;

	status.cuda = cudaDeviceSynchronize();
	if (status.failed()) return status;


	cudaFree(out1);
	cudaFree(out3);


	if (mostrar) printInfo(gen, eval);

	return status;

}


ErrorInfo evaluate(bool* pop, size_t POP_SIZE, int length, float* dev_fit, EvalInfo& eval, float* W, float* G, int gen, float MAX_WEIGHT) {

	cudaEvent_t startFitness, stopFitness;
	if (KERNEL_TIMING) {
		cudaEventCreate(&startFitness);
		cudaEventCreate(&stopFitness);

		cudaEventRecord(startFitness);
	} // KERNEL_TIMING

	fitness_knapsack << < (unsigned int)POP_SIZE, MAX_THREADS_PER_BLOCK >> > (pop, dev_fit, length, W, G, MAX_WEIGHT, PENAL);
	cudaDeviceSynchronize();
	if (KERNEL_TIMING) {
		cudaEventRecord(stopFitness);
		cudaEventSynchronize(stopFitness);
		float milisecsFitness = 0;
		cudaEventElapsedTime(&milisecsFitness, startFitness, stopFitness);
		timeFitness += milisecsFitness;
	}  // KERNEL_TIMING

	return evaluate_(POP_SIZE, dev_fit, eval,gen);
}


ErrorInfo evaluate_bitwise(Data* pop, size_t POP_SIZE, int realLength,int length, float* dev_fit, EvalInfo& eval,float* W, float* G,int gen,float MAX_WEIGHT) {
	cudaEvent_t startFitness, stopFitness;
	if (KERNEL_TIMING) {
		cudaEventCreate(&startFitness);
		cudaEventCreate(&stopFitness);

		cudaEventRecord(startFitness);
	}  // KERNEL_TIMING
	fitness_knapsack_b << < (unsigned int)POP_SIZE, MAX_THREADS_PER_BLOCK >> > (pop, dev_fit, length,realLength, FirstBitMask, W, G, MAX_WEIGHT, PENAL);
	cudaDeviceSynchronize();
	if (KERNEL_TIMING) {
		cudaEventRecord(stopFitness);
		cudaEventSynchronize(stopFitness);

		float milisecsFitness = 0;
		cudaEventElapsedTime(&milisecsFitness, startFitness, stopFitness);
		timeFitness += milisecsFitness;
	}
	return evaluate_(POP_SIZE, dev_fit, eval,gen);

}


__global__ void WG_Fijos(float* W, float* G, int len) {
	unsigned int thIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thIdx < len) {
		W[thIdx] = thIdx % 2 + 1;
		G[thIdx] = thIdx % 10;
	}

}


void inicializarWG(float** W, float** G, int len) {
	cudaMalloc(W, len * sizeof(float));
	cudaMalloc(G, len * sizeof(float));
	int blocks = (len  + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
	WG_Fijos << <blocks, MAX_THREADS_PER_BLOCK >> > (*W, *G, len);


}

void generarAleatorioPacket(curandGenerator_t& generator, size_t bytes, void* buffer) {
	size_t N = (3 + bytes) / 4;
	unsigned int *ptr = reinterpret_cast<unsigned int*>(buffer);
	curandGenerate(generator, ptr, N);
}



ErrorInfo GA(size_t POP_SIZE,int len,int iters,bool dpx_cross,float crossProb,float mutProb,	unsigned long long seed, float MAX_WEIGHT) {
	/*cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);*/

	unsigned int NRO_BLOCKS = ((unsigned int)POP_SIZE + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
	double max_fitness;
	int gen_max_fitness = 0;
	if (dpx_cross) printf("DPX ");
	else printf("SPX ");

	printf("bool POP_SIZE=%u length=%d seed=%llu pCross=%.2f pMut=%.2f\n",  (unsigned int)POP_SIZE, len, seed, crossProb, mutProb);
	ErrorInfo status;
	bool *pop, *npop;
	float* fit;
	int* win; // indices de individuos ganadores en el tournment
	int* tourn;
	float  *probs;
	int *points;


	float *W;
	float *G;
	inicializarWG(&W, &G, len);

	EvalInfo eval;
	status.cuda = InitFit(&fit, POP_SIZE);
	status.cuda = InitWin(&win, POP_SIZE);
	status.cuda = InitTournRandom(&tourn, POP_SIZE);
	status = initProbs(&probs, &points, POP_SIZE);

	curandGenerator_t generator;
	status.curand = initGenerator(generator, seed);
	if (status.failed()) return status;

	// usa la curand device API para generar la poblacion sin prealocar numeros aleatorios para eso
	status = generatePOP_device(seed, POP_SIZE, len, &pop, &npop);

	// cambia el offset del generador para que no se sobreponga con el usado para la generacion de la poblacion
	curandSetGeneratorOffset(generator, POP_SIZE * len);

	if (status.failed()) {
		fprintf(stderr, "generatePOP failed!");
		return status;
	}

	int gen = 0;
	status = evaluate(pop, POP_SIZE, len,fit,eval,W,G,gen,MAX_WEIGHT);
	max_fitness = eval.max;


	for ( gen = 1; gen <= iters; gen++) { // while not optimalSolutionFound
		// elegir POP_SIZE parejas para el torneo
		status = makeRandomNumbersTournement(generator, POP_SIZE, tourn);
		if (status.failed()) return status;

		// elegir POP_SIZE ganadores
		tournament<<< NRO_BLOCKS , MAX_THREADS_PER_BLOCK>>> (fit, tourn, win,POP_SIZE);
		status.cuda = cudaGetLastError();
		if (status.failed()) return status;

		// seleccion
		if (dpx_cross) {
			makeRandomNumbersDpx(generator, POP_SIZE, len, probs, points);
		}
		else {
			makeRandomNumbersSpx(generator, POP_SIZE, len, probs, points);
		}

		cudaEvent_t startCross, stopCross;
		if (KERNEL_TIMING) {
			cudaEventCreate(&startCross);
			cudaEventCreate(&stopCross);

			cudaEventRecord(startCross);
		} 	 // KERNEL_TIMING
		
		if (dpx_cross){
			dpx << < POP_SIZE / 2, MAX_THREADS_PER_BLOCK >> >(pop, npop, win, probs, points, len, crossProb);
		}
		else {
			spx << < POP_SIZE / 2, MAX_THREADS_PER_BLOCK >> >(pop, npop, win, probs, points, len, crossProb);
		}
		status.cuda = cudaGetLastError();
		if (status.failed()) return status;

		status.cuda = cudaDeviceSynchronize();
		if (status.failed()) return status;
		if (KERNEL_TIMING) {
			cudaEventRecord(stopCross);
			cudaEventSynchronize(stopCross);
			float milisecs = 0;
			cudaEventElapsedTime(&milisecs, startCross, stopCross);
			timeCross += milisecs;
		} // KERNEL_TIMING


		// elegir numeros aleatorios para mutacion 
		// se reusa la memoria que se uso para los numeros aleatorios de la seleccion
		status = makeRandomNumbersMutation(generator, POP_SIZE, len, probs, points);

		// mutacion
		mutation << < NRO_BLOCKS, MAX_THREADS_PER_BLOCK >> >(npop, probs, points, len, mutProb,POP_SIZE);
		status.cuda = cudaGetLastError();
		if (status.failed()) return status;
		status.cuda = cudaDeviceSynchronize();

		bool* tmp;
		tmp = pop;
		pop = npop;
		npop = tmp;
		status = evaluate(pop, POP_SIZE, len, fit, eval, W, G,gen, MAX_WEIGHT);
		
		if (eval.max > max_fitness) {
			gen_max_fitness = gen;
			max_fitness = eval.max;
		}
	}
	printf("Gen. max fitness: %d (%f)\n", gen_max_fitness, max_fitness);
	return status;
}

ErrorInfo GA_bitwise(size_t POP_SIZE, int len, int iters, bool dpx_cross, float crossProb, float mutProb,unsigned long long seed, float MAX_WEIGHT) {

	unsigned int NRO_BLOCKS = ((unsigned int)POP_SIZE + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
	if (dpx_cross) printf("DPX "); 
	else printf("SPX ");
	printf("bitwise(%u) POP_SIZE=%u length=%d seed=%llu pCross=%.2f pMut=%.2f\n",(unsigned int)(sizeof(Data) * 8), (unsigned int)POP_SIZE, len, seed,crossProb,mutProb);
	ErrorInfo status;
	Data *pop, *npop;
	float* fit;
	int* win;  // indices de individuos ganadores en el tournment
	int* tourn;
	float  *probs;
	int *points;

	double max_fitness;
	int gen_max_fitness = 0;

	float *W;
	float *G;
	inicializarWG(&W, &G, len);


	EvalInfo eval;
	int realLength = (len + DataSize - 1) / DataSize;
	status.cuda = InitFit(&fit, POP_SIZE);
	status.cuda = InitWin(&win, POP_SIZE);
	status.cuda = InitTournRandom(&tourn, POP_SIZE);
	status = initProbs(&probs, &points, POP_SIZE);

	curandGenerator_t generator;
	status.curand = initGenerator(generator, seed);
	if (status.failed()) return status;

	// usa la curand device API para generar la poblacion sin prealocar numeros aleatorios para eso
	status = generatePOP_device_bitwise(seed, POP_SIZE, len, &pop, &npop);

	// cambia el offset del generador para que no se sobreponga con el usado para la generacion de la poblacion
	curandSetGeneratorOffset(generator, POP_SIZE * len);

	if (status.failed()) {
		fprintf(stderr, "generatePOP failed!");
		return status;
	}

	int gen = 0;
	status = evaluate_bitwise(pop, POP_SIZE, realLength,len, fit, eval,W,G,gen, MAX_WEIGHT);
	max_fitness = eval.max;

	for ( gen = 1; gen <= iters; gen++) { // while not optimalSolutionFound
											 // elegir POP_SIZE parejas para el torneo
		status = makeRandomNumbersTournement(generator, POP_SIZE, tourn);
		if (status.failed()) return status;

		// elegir POP_SIZE ganadores
		tournament << < NRO_BLOCKS, MAX_THREADS_PER_BLOCK >> > (fit, tourn, win,POP_SIZE);
		status.cuda = cudaGetLastError();
		if (status.failed()) return status;

		// seleccion
		if (dpx_cross) {
			makeRandomNumbersDpx(generator, POP_SIZE, len, probs, points);
		}
		else {
			makeRandomNumbersSpx(generator, POP_SIZE, len, probs, points);
		}
		cudaEvent_t startCross, stopCross;
		if (KERNEL_TIMING) {
			
			cudaEventCreate(&startCross);
			cudaEventCreate(&stopCross);

			cudaEventRecord(startCross);
		} //KERNEL_TIMING
		if (dpx_cross) {
			dpx_b << < POP_SIZE / 2, MAX_THREADS_PER_BLOCK >> >(pop, npop, win, probs, points, realLength, crossProb);
		}
		else {
			spx_b << < POP_SIZE / 2, MAX_THREADS_PER_BLOCK >> >(pop, npop, win, probs, points, realLength, crossProb);
		}
		status.cuda = cudaGetLastError();
		if (status.failed()) return status;

		status.cuda = cudaDeviceSynchronize();
		if (status.failed()) return status;
		if (KERNEL_TIMING) {
			cudaEventRecord(stopCross);
			cudaEventSynchronize(stopCross);
			float milisecs = 0;
			cudaEventElapsedTime(&milisecs, startCross, stopCross);
			timeCross += milisecs;
		} //KERNEL_TIMING

		// elegir numeros aleatorios para mutacion 
		// se reusa la memoria que se uso para los numeros aleatorios de la seleccion
		status = makeRandomNumbersMutation(generator, POP_SIZE, len, probs, points);

		// mutacion
		mutation_b << < NRO_BLOCKS, MAX_THREADS_PER_BLOCK >> >(npop, probs, points, realLength, FirstBitMask, mutProb,POP_SIZE);
		status.cuda = cudaGetLastError();
		if (status.failed()) return status;
		status.cuda = cudaDeviceSynchronize();

		Data* tmp;
		tmp = pop;
		pop = npop;
		npop = tmp;
		status  = evaluate_bitwise(pop, POP_SIZE, realLength, len, fit, eval, W, G,gen, MAX_WEIGHT);

		if (eval.max > max_fitness) {
			gen_max_fitness = gen;
			max_fitness = eval.max;
		}
	}
	printf("Gen. max fitness: %d (%f)\n", gen_max_fitness, max_fitness);
	return status;
}





void setArgumentsFromCmd(int argc, char** argv,float& pMutacion, float& pCruce, float& MAX_WEIGHT, unsigned int& POP_SIZE,int& length, int& iters, unsigned long long& seed,bool& use_dpx,bool& bitwise) {
	// valores por defecto
	POP_SIZE = 2048;
	length = 10000;
	iters = 10000;
	pMutacion = 0.4f;
	pCruce = 0.9f;
	seed = 2825521;
	use_dpx = false;
	bitwise = true;

	bool relativeW = true;
	float w = 0.1f;
	for (int i = 1; i < argc; i++) {
		if (strcmp("-k", argv[i]) == 0) {
			KERNEL_TIMING = true; // -k = Activar KERNEL_TIMING 
		}
		if (strcmp("-dpx", argv[i]) == 0) {
			use_dpx = true;
		}
		if (strcmp("-u", argv[i]) == 0) {
			UNIT_TEST = true;
		}
		if (strcmp("-nomax", argv[i]) == 0) {
			MAX_DETALLADO = false;
		}
		if (strcmp("-noteo", argv[i]) == 0) {
			CALC_MAXFIT = false;
		}
		if (strcmp("-bool", argv[i]) == 0) {
			bitwise = false;
		}
		if (strcmp("-o", argv[i]) == 0) {
			if (i + 1 < argc) SALIDA_STEP = atoi(argv[i + 1]);
		}

		if (strcmp("-m", argv[i]) == 0) {
			if (i + 1 < argc) pMutacion = (float)(atof(argv[i + 1]) / 100.0); // porcentaje
		}
		if (strcmp("-x", argv[i]) == 0) {
			if (i + 1 < argc) pCruce = (float)(atof(argv[i + 1]) / 100.0);  // porcentaje
		}
		if (strcmp("-w", argv[i]) == 0) {
			if (i + 1 < argc) w = (float)(atof(argv[i + 1]) ); // maximo peso = proporcion de longitud
		}
		if (strcmp("-W", argv[i]) == 0) {
			if (i + 1 < argc) {
				w = (float)(atof(argv[i + 1])); // maximo peso absoluto (ignora -w)
				relativeW = false;
			}
		}
		if (strcmp("-len", argv[i]) == 0) {
			if (i + 1 < argc) length = (atoi(argv[i + 1])); // longitud del individuo
		}
		if (strcmp("-p", argv[i]) == 0) {
			if (i + 1 < argc) POP_SIZE = (unsigned int)(atoi(argv[i + 1])); // longitud del individuo
		}
		if (strcmp("-i", argv[i]) == 0) {
			if (i + 1 < argc) iters = (atoi(argv[i + 1])); // iteraciones
		}
		if (strcmp("-s", argv[i]) == 0) {
			if (i + 1 < argc) seed = (unsigned long long)atoll(argv[i + 1]);
		}

	}
	if (relativeW) {
		MAX_WEIGHT = w * length;
	}
}


// A utility function that returns maximum of two integers
int max_(int a, int b) { return (a > b) ? a : b; }

// http://www.geeksforgeeks.org/knapsack-problem/
// Returns the maximum value that can be put in a knapsack of capacity W
int knapSack(int W, int wt[], int val[], int n)
{
	int i, w;
	int** K = new int*[n + 1];
	for (int i = 0; i <= n; i++) {
		K[i] = new int[W + 1];
	}

	// Build table K[][] in bottom up manner
	for (i = 0; i <= n; i++)
	{
		for (w = 0; w <= W; w++)
		{
			if (i == 0 || w == 0)
				K[i][w] = 0;
			else if (wt[i - 1] <= w)
				K[i][w] = max_(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w]);
			else
				K[i][w] = K[i - 1][w];
		}
	}

	int res =  K[n][W];
	for (int i = 0; i <= n; i++) {
		delete[] K[i];
	}
	delete[] K;
	return res;
}

// calcular solucion del problema usando programacion dinamica (en CPU)
int solveKnapsack(int W, int n) {
	int* wt = new int[n];
	int* val = new int[n];
	for (int i = 0; i < n; i++) {
		wt[i] = i % 2 + 1;
		val[i] = i % 10;
	}
	int res = knapSack(W, wt, val, n);
	delete[] wt;
	delete[] val;
	return res;
}


void PU_DPX(); // test.cu


int main(int argc, char** argv)
{


	cudaError_t cudaStatus;
	ErrorInfo status;



	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	
	std::clock_t c_start = std::clock();

	timeFitness = timeCross = 0;

	// TODO: arreglar para POP_SIZE no multiplo de MAX_THREADS
	unsigned int POP_SIZE = 2048;
	 // peso maximo de la mochila
	int len,iters ;
	float MAX_WEIGHT, pMutacion, pCruce;
	unsigned long long seed ;
	bool use_dpx,bitwise ;

	setArgumentsFromCmd(argc, argv, pMutacion, pCruce, MAX_WEIGHT, POP_SIZE, len, iters, seed, use_dpx,bitwise);

	if (UNIT_TEST) {
		PU_DPX();
		exit(0);
	}

	if (bitwise) {
		GA_bitwise(POP_SIZE, len, iters, use_dpx, pCruce, pMutacion, seed, MAX_WEIGHT);
	}
	else {
		GA(POP_SIZE, len, iters, use_dpx, pCruce, pMutacion, seed, MAX_WEIGHT);
	}
	
	std::clock_t c_end = std::clock();
	double time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	printf("Tiempo total: %.3fs", time_elapsed_ms / 1000.0);
	if (CALC_MAXFIT) {
		int teorico = solveKnapsack((int)MAX_WEIGHT, len);
		printf(".     Max posible fitness : %i", teorico);
	}
	printf("\n");
	if (KERNEL_TIMING) {
		printf("Tiempo fitness: %.3fs, Tiempo cross: %.3fs\n", timeFitness / 1000, timeCross / 1000);
	}  //KERNEL_TIMING 

	

	//std::cout << "Press any key to exit . . .";
	//std::cin.get();
	return 0;
}


