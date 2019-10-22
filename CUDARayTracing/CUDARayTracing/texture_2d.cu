#include "cuda_runtime.h" //for intellisense
#include "device_launch_parameters.h" //for intellisense

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <DirectXMath.h>

#define PI 3.1415926536f

struct qwe {
	int a;
	int b;
	DirectX::XMFLOAT3 c;

	__device__ void Test() {
		a++;
	}
};

__global__ void cuda_kernel_texture_2d(qwe q,unsigned char* surface, int width, int height, size_t pitch, float* spheres, int num_sphere)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float* pixel;

	if (x >= width || y >= height) return;

	// get a pointer to the pixel at (x,y)
	pixel = (float*)(surface + y * pitch) + 4 * x;

	pixel[0] = 0.0 * x / width;
	pixel[1] = 0.0 * y / height; // green
	pixel[2] = spheres[3]; // blue
	pixel[3] = 1; // alpha

	q.Test();
	pixel[0] = q.c.x;
}

extern "C"
void cuda_texture_2d(void* surface, int width, int height, size_t pitch, float* spheres, int num_sphere)
{
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

	qwe q;
	q.a = 1;
	q.b = 1;
	q.c = { 1,0,1 };

	cuda_kernel_texture_2d << <Dg, Db >> > (q, (unsigned char*)surface, width, height, pitch, spheres, num_sphere);

	error = cudaGetLastError();

	if (error != cudaSuccess)
	{
		printf("cuda_kernel_texture_2d() failed to launch error = %d\n", error);
	}
}
