#include "stdint.h"
#include "ErrorInfo.h"

// The datatype Data should be defined as unsigned char, unsigned short, unsigned int or unsigned long.
typedef unsigned char Data;
// The constant DataSize should be defined as 8 (if Data is unsigned char), 16 (if Data is unsigned short), 
// 32 (if Data is unsigned int) or 64 (if Data is unsigned long).
#define DataSize 8


#define DataMask (Data)(~(0ull))
#define FirstBitMask (Data)(1ull << (DataSize - 1))

ErrorInfo generatePOP_device_bitwise(unsigned long seed, size_t POP_SIZE, int len, Data** pop, Data** npop);
__global__ void tournament_b(int * fit, int * random, int * win);
__global__ void mutation_b(Data *pop, float *randomPM, int *randomPoint, int length, Data mask, float PROB_MUT);
__global__ void dpx_b(Data *pop, Data *npop, int *pos, float *randomPC, int *randomPoint, int length, float PROB_CROSS);
__global__ void spx_b(Data *pop, Data *npop, int *pos, float *randomPC, int *randomPoint, int length, float PROB_CROSS);
__global__ void fitness_b(Data * pop, int * fit, int realLength, Data mask, int length);