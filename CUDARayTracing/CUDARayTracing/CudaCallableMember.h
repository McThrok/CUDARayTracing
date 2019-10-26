#pragma once

#ifdef __CUDACC__
#define CUDA __host__ __device__
#else
#define CUDA
#endif 