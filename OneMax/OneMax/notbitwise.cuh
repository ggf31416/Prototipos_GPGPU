#include "stdint.h"
#include "ErrorInfo.h"


__global__ void mutation(bool *pop, float *randomPM, int *randomPoint, int length, float PROB_MUT);
__global__ void fitness(bool * pop, int * fit, int length);
__global__ void dpx(bool *pop, bool *npop, int *pos, float *randomPC, int *randomPoint, int length, float PROB_CROSS);
__global__ void spx(bool *pop, bool *npop, int *pos, float *randomPC, int *randomPoint, int length, float PROB_CROSS);
__global__ void tournament(int * fit, int * random, int * win);
ErrorInfo generatePOP_device(unsigned long seed, size_t POP_SIZE, int len, bool** pop, bool** npop);