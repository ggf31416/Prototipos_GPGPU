#pragma once

#include "curand.h"
#include <stdio.h>
#include "cuda.h"
#include "defines_onemax.h"

static const char *curandGetErrorString(curandStatus_t error)
{
	switch (error)
	{
	case CURAND_STATUS_SUCCESS:
		return "CURAND_STATUS_SUCCESS";

	case CURAND_STATUS_VERSION_MISMATCH:
		return "CURAND_STATUS_VERSION_MISMATCH";

	case CURAND_STATUS_NOT_INITIALIZED:
		return "CURAND_STATUS_NOT_INITIALIZED";

	case CURAND_STATUS_ALLOCATION_FAILED:
		return "CURAND_STATUS_ALLOCATION_FAILED";

	case CURAND_STATUS_TYPE_ERROR:
		return "CURAND_STATUS_TYPE_ERROR";

	case CURAND_STATUS_OUT_OF_RANGE:
		return "CURAND_STATUS_OUT_OF_RANGE";

	case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
		return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

	case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
		return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

	case CURAND_STATUS_LAUNCH_FAILURE:
		return "CURAND_STATUS_LAUNCH_FAILURE";

	case CURAND_STATUS_PREEXISTING_FAILURE:
		return "CURAND_STATUS_PREEXISTING_FAILURE";

	case CURAND_STATUS_INITIALIZATION_FAILED:
		return "CURAND_STATUS_INITIALIZATION_FAILED";

	case CURAND_STATUS_ARCH_MISMATCH:
		return "CURAND_STATUS_ARCH_MISMATCH";

	case CURAND_STATUS_INTERNAL_ERROR:
		return "CURAND_STATUS_INTERNAL_ERROR";
	}

	return "<unknown>";
}

struct ErrorInfo {
	cudaError_t cuda;
	curandStatus_t curand;

	ErrorInfo() : cuda(cudaSuccess), curand(CURAND_STATUS_SUCCESS) {}

	bool ok() { return (cudaSuccess == this->cuda) && (CURAND_STATUS_SUCCESS == this->curand); }
	bool failed() {
		if ((cudaSuccess == this->cuda) && (CURAND_STATUS_SUCCESS == this->curand)) {
			return false;
		}
		else {
			if (cudaSuccess != this->cuda) {
				printf("Error Cuda: %s\n", cudaGetErrorString(this->cuda));
			}
			if (CURAND_STATUS_SUCCESS != this->curand) {
				printf("Error Curand: %s (%d)\n", curandGetErrorString(this->curand), (int)this->curand);
			}
			return true;
		}
	}
};
