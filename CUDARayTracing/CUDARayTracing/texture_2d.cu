#include "cuda_runtime.h" //for intellisense
#include "device_launch_parameters.h" //for intellisense

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vec3.h"
#include "Sphere.h"
#include "Scene.h"
#include "Screen.h"
#include <algorithm>

#define PI 3.1415926536f


__global__ void cuda_kernel_texture_2d(Screen screen, Scene scene)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float* pixel;

	if (x >= screen.width || y >= screen.height) return;

	// get a pointer to the pixel at (x,y)
	pixel = (float*)((char*)screen.surface + y * screen.pitch) + 4 * x;

	Ray ray = scene.cam.CastScreenRay(x, y);

	pixel[3] = 1.0f; // alpha
	Sphere sphere = scene.spheres[0];
	float dist = sphere.findIntersection(ray);
	if (dist > 0)
	{
		vec3 p = ray.getPointAt(dist);
		vec3 col = scene.light.getColor(scene.cam, p, sphere.getNormalAt(p),sphere.color);
		pixel[0] = col.x;// 0.0 * x / width;
		pixel[1] = col.y;// 0.0 * y / height; // green
		pixel[2] = col.z; // blue
	}
	else {
		pixel[0] = 0.0f;// 0.0 * x / width;
		pixel[1] = 1.0f;// 0.0 * y / height; // green
		pixel[2] = 0.0f; // blue

	}

}

extern "C"
void cuda_texture_2d(Screen screen, Scene scene)
{
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((screen.width + Db.x - 1) / Db.x, (screen.width + Db.y - 1) / Db.y);

	cuda_kernel_texture_2d << <Dg, Db >> > (screen, scene);

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
	color = colors + y + x * height * 4;

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


	cuda_kernel_copy_colors << <Dg, Db >> > ((unsigned char*)surface, width, height, pitch, colors);

	error = cudaGetLastError();

	if (error != cudaSuccess)
	{
		printf("cuda_kernel_texture_2d() failed to launch error = %d\n", error);
	}
}
