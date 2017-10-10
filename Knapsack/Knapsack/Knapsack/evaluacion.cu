#include "evaluacion.cuh"

template<typename T>
__global__ void sumar(T* dev_rnd, float* dev_output, unsigned int len)
{
	__shared__ float intermedio[MAX_THREADS_PER_BLOCK];
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tid = threadIdx.x;
	intermedio[threadIdx.x] = idx < len ? dev_rnd[idx] : 0;
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

__global__ void minimo(float* dev_rnd, float* dev_output, unsigned int len)
{
	__shared__ float intermedio[MAX_THREADS_PER_BLOCK];
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

__global__ void maximo(float* dev_rnd, float* dev_output, unsigned int len)
{
	__shared__ float intermedio[MAX_THREADS_PER_BLOCK];
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

__global__ void contarInvalidos(float* fitness, float* dev_output, size_t pop_size) {
	__shared__ float intermedio[MAX_THREADS_PER_BLOCK];
	unsigned int tid = threadIdx.x;
	intermedio[tid] = 0;
	for (int i = tid; i < pop_size; i += MAX_THREADS_PER_BLOCK) {
		intermedio[tid] += (fitness[i] < 0) ? 1 : 0;
	}
	// reduction algorithm to add the partial fitness values
	__syncthreads();
	int i = MAX_THREADS_PER_BLOCK / 2;
	while (i != 0) {
		if (threadIdx.x < i)
			intermedio[threadIdx.x] = intermedio[threadIdx.x] + intermedio[threadIdx.x + i];
		__syncthreads();
		i = i / 2;
	}

	// finally thread 0 writes the fitness value in global memory
	if (threadIdx.x == 0)
		dev_output[blockIdx.x] = intermedio[0];
}

__global__ void sumarValidez(float* fitness, float* dev_output, size_t pop_size) {
	__shared__ float intermedio_val[MAX_THREADS_PER_BLOCK];
	__shared__ float intermedio_inval[MAX_THREADS_PER_BLOCK];
	__shared__ float intermedio_cant[MAX_THREADS_PER_BLOCK];
	unsigned int tid = threadIdx.x;
	intermedio_val[tid] = 0;
	intermedio_inval[tid] = 0;
	intermedio_cant[tid] = 0;
	for (int i = tid; i < pop_size; i += MAX_THREADS_PER_BLOCK) {
		if (fitness[i] >= 0) {
			intermedio_val[tid] += fitness[i];
		}
		else {
			intermedio_inval[tid] += fitness[i];
			intermedio_cant[tid] += 1;
		}

	}
	// reduction algorithm to add the partial fitness values
	__syncthreads();
	int i = MAX_THREADS_PER_BLOCK / 2;
	while (i != 0) {
		if (threadIdx.x < i)
			intermedio_val[threadIdx.x] = intermedio_val[threadIdx.x] + intermedio_val[threadIdx.x + i];
		__syncthreads();
		i = i / 2;
	}

	// reduction algorithm to add the partial fitness values
	__syncthreads();
	i = MAX_THREADS_PER_BLOCK / 2;
	while (i != 0) {
		if (threadIdx.x < i)
			intermedio_inval[threadIdx.x] = intermedio_inval[threadIdx.x] + intermedio_inval[threadIdx.x + i];
		__syncthreads();
		i = i / 2;
	}

	// reduction algorithm to add the partial fitness values
	__syncthreads();
	i = MAX_THREADS_PER_BLOCK / 2;
	while (i != 0) {
		if (threadIdx.x < i)
			intermedio_cant[threadIdx.x] = intermedio_cant[threadIdx.x] + intermedio_cant[threadIdx.x + i];
		__syncthreads();
		i = i / 2;
	}

	// finally thread 0 writes the fitness value in global memory
	if (threadIdx.x == 0) {
		dev_output[0] = intermedio_cant[0]; // cant invalidos
		dev_output[1] = intermedio_inval[0]; // suma invalidos 
		dev_output[2] = intermedio_val[0]; // suma validos

	}

}

__global__ void minimoValidos(float* fitness, float* dev_output, size_t pop_size) {
	__shared__ float intermedio[MAX_THREADS_PER_BLOCK];
	unsigned int tid = threadIdx.x;
	intermedio[tid] = 0;
	for (int i = tid; i < pop_size; i += MAX_THREADS_PER_BLOCK) {
		if (fitness[i] > 0) {
			intermedio[tid] = intermedio[tid] > 0 ? min(intermedio[tid], fitness[i]) : fitness[i];
		}
	}
	// reduction algorithm to add the partial fitness values
	__syncthreads();
	int i = MAX_THREADS_PER_BLOCK / 2;
	while (i != 0) {
		if (threadIdx.x < i) {
			if (intermedio[threadIdx.x] > 0 && intermedio[threadIdx.x + i] > 0) {
				// el menor de los valores > 0
				intermedio[threadIdx.x] = min(intermedio[threadIdx.x], intermedio[threadIdx.x + i]);
			}
			else {
				// el mayor de los dos valores (el mayor a 0 o 0 si los 2 son 0)
				intermedio[threadIdx.x] = max(intermedio[threadIdx.x], intermedio[threadIdx.x + i]);
			}

		}
		__syncthreads();
		i = i / 2;
	}

	// finally thread 0 writes the fitness value in global memory
	if (threadIdx.x == 0)
		dev_output[blockIdx.x] = intermedio[0];
}

