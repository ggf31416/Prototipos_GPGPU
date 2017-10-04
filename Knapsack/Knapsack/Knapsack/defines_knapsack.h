#pragma once
#include "stdint.h"

#define INIT_THREADS 128
#define MAX_THREADS_PER_BLOCK 512

#define PENAL 1 // valor penalizacion sobrepeso
#define T_FIT float // tipo de datos de fitness

#define SALIDA 1

#define ORDEN_DPX 1 // forzar pnt2 > pnt 1?

// para bitwise
// The constant DataSize should be defined as 8 (if Data is unsigned char), 16 (if Data is unsigned short), 
// 32 (if Data is unsigned int) or 64 (if Data is unsigned long).
#define DataSize 32

// The datatype Data should be defined as unsigned char, unsigned short, unsigned int or unsigned long.
#if DataSize == 8
typedef unsigned char Data;

#elif DataSize == 16
typedef uint16_t Data;

#elif DataSize == 32
typedef uint32_t Data;

#else 
typedef uint64_t Data;

#endif // DataSize == 8

