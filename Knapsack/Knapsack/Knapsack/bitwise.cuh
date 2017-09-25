#include "stdint.h"
#include "ErrorInfo.h"
#include "defines_knapsack.h"

#define DataMask (Data)(~(0ull))
#define FirstBitMask (Data)(1ull << (DataSize - 1))

ErrorInfo generatePOP_device_bitwise(unsigned long seed, size_t POP_SIZE, int len, Data** pop, Data** npop);
__global__ void tournament_b(float * fit, int * random, int * win);
__global__ void mutation_b(Data *pop, float *randomPM, int *randomPoint, int length, Data mask, float PROB_MUT);
__global__ void dpx_b(Data *pop, Data *npop, int *pos, float *randomPC, int *randomPoint, int length, float PROB_CROSS);
__global__ void spx_b(Data *pop, Data *npop, int *pos, float *randomPC, int *randomPoint, int length, float PROB_CROSS);
__global__ void fitness_knapsack_b(Data * pop, float* fit, int length, int realLength, Data mask, float* W, float* G, float MAX_W, float P);