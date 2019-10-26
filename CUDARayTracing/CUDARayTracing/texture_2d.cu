#include "cuda_runtime.h" //for intellisense
#include "device_launch_parameters.h" //for intellisense

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vec3.h"

#define PI 3.1415926536f


struct qwe {
	int a;
	int b;
	vec3 c;

	__device__ void Test() {
		a++;
	}

};

__global__ void cuda_kernel_texture_2d(qwe q, unsigned char* surface, int width, int height, size_t pitch, float* spheres, int num_sphere)
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
	vec3 a = vec3();
	a.cross(a);
	a.dot(a + 1);

	//float c = a.length();

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


__global__ void cuda_kernel_copy_colors(unsigned char* surface, int width, int height, size_t pitch, float* colors)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float* pixel;
	float* color;

	if (x >= width || y >= height) return;

	// get a pointer to the pixel at (x,y)
	pixel = (float*)(surface + y * pitch) + 4 * x;
	color = colors + y + x * height*4;

	pixel[0] = color[0];
	pixel[1] = color[1];
	pixel[2] = color[2];
	pixel[3] = color[3];
}

extern "C"
void cuda_copy_colors(void* surface, int width, int height, size_t pitch, float* colors)
{
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);


	cuda_kernel_copy_colors << <Dg, Db >> > ( (unsigned char*)surface, width, height, pitch, colors);

	error = cudaGetLastError();

	if (error != cudaSuccess)
	{
		printf("cuda_kernel_texture_2d() failed to launch error = %d\n", error);
	}
}
