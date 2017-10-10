#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include <device_functions.h>
#include "device_launch_parameters.h"
#include "defines_knapsack.h"

__global__ void maximo(float* dev_rnd, float* dev_output, unsigned int len);
__global__ void sumarValidez(float* fitness, float* dev_output, size_t pop_size);
__global__ void minimoValidos(float* fitness, float* dev_output, size_t pop_size);