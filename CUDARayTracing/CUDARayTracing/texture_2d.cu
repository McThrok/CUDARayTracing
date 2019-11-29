#include "cuda_runtime.h" //for intellisense
#include "device_launch_parameters.h" //for intellisense

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "vec3.h"
#include "Scene.h"
#include "Screen.h"
#include <algorithm>

#define PI 3.1415926536f
#define FULL_MASK 0xffffffff

#define DIFFUSE 0.8
#define AMBIENT 0.2


__device__ vec3 getSphereColor(vec3& position, vec3& color, vec3& point) {
	vec3 lightPos = vec3(0, 15, 25);
	vec3 n = point - position;

	vec3 toLight = (lightPos - point).norm();
	float diff = fmaxf(toLight.dot(n), 0);
	return color * (diff * DIFFUSE + AMBIENT);
}

__device__ vec3 castScreenRay(CameraData& c, int& x, int& y, int& width, int& height) {
	//r=1
	float xAngle = c.fov * (1.0f * x / width - 0.5f);
	float yAngle = c.fov * (1.0f * y / height - 0.5f);

	return vec3(c.forward + c.right * sinf(xAngle) * c.aspect + c.up * sinf(yAngle)).norm();
}

__device__ float findIntersection(vec3& pos, float& r, vec3& rayOrigin, vec3& rayDir) {
	vec3 v(rayOrigin.x - pos.x, rayOrigin.y - pos.y, rayOrigin.z - pos.z);
	float b = (2 * v.x * rayDir.x) + (2 * v.y * rayDir.y) + (2 * v.z * rayDir.z);
	float c = v.dot(v) - (r * r);

	float dist = -1;
	float discriminant = b * b - 4 * c;
	if (discriminant >= 0) {
		float tmp = sqrtf(discriminant);
		dist = ((-b - tmp) / 2) - 0.000001f;
		if (dist < 0)
			dist = ((-b + tmp) / 2) - 0.000001f;
	}

	return dist;
}
__global__ void cuda_kernel_texture_2dx(Screen screen, Scene scene)
{
	int si = threadIdx.x;
	int x = threadIdx.y + blockDim.y * blockIdx.x;
	int y = blockIdx.y;
	int i;
	float d;

	if (x >= screen.width || y >= screen.height) return;

	float* pixel = (float*)((char*)screen.surface + y * screen.pitch) + 4 * x;

	vec3 rayDir = castScreenRay(scene.cam, x, y, screen.width, screen.height);
	vec3 rayOrigin = scene.cam.position;


	int index = -1;
	float dist = 1000;

	for (i = 0; i < 16; i++)
	{
		int idx = si + 32 * i;
		d = findIntersection(scene.position[idx], scene.radius[idx], rayOrigin, rayDir);
		if (d > 0)
		{
			dist = d;
			index = idx;
		}
	}

	//if (__any_sync(FULL_MASK, dist > 0))
	//{
	//	for (int offset = 1; offset < 32; offset <<= 1) {
	//		d = __shfl_down_sync(0xFFFFFFFF, dist, offset, 32);
	//		i = __shfl_down_sync(0xFFFFFFFF, index, offset, 32);
	//		if (dist < 0 || (d > -1 && d < dist))
	//		{
	//			dist = d;
	//			index = i;
	//		}
	//	}
	//}

	if (si == 0)
	{
		if (index >= 0)
		{
			vec3 p = rayOrigin + rayDir * dist;
			vec3 col = getSphereColor(scene.position[index], scene.color[index], p);

			pixel[0] = col.x;
			pixel[1] = col.y;
			pixel[2] = col.z;
			pixel[3] = 1.0f;
		}
		else
		{
			pixel[0] = 0.0f;
			pixel[1] = 1.0f;
			pixel[2] = 0.0f;
			pixel[3] = 1.0f;
		}
	}
}

extern "C"
void cuda_texture_2dx(Screen screen, Scene scene)
{
	cudaError_t error = cudaSuccess;

	dim3 blockSize = dim3(32, 16, 1);
	dim3 gridSize = dim3(screen.width / blockSize.y, screen.height, 1);
	//dim3 gridSize = dim3(screen.width / blockSize.y, 128, 1);

	cuda_kernel_texture_2dx << < gridSize, blockSize >> > (screen, scene);

	error = cudaGetLastError();

	if (error != cudaSuccess)
	{
		printf("cuda_kernel_texture_2d() failed to launch error = %d\n", error);
	}
}

__global__ void cuda_kernel_texture_2d(Screen screen, Scene scene)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float* pixel;

	if (x >= screen.width || y >= screen.height) return;

	// get a pointer to the pixel at (x,y)
	pixel = (float*)((char*)screen.surface + y * screen.pitch) + 4 * x;

	vec3 rayDir = castScreenRay(scene.cam, x, y, screen.width, screen.height);
	vec3 rayOrigin = scene.cam.position;

	pixel[3] = 1.0f; // alpha

	float minDist = 10000;
	int idx = -1;

	for (int i = 0; i < scene.sphere_num; i++)
	{
		float dist = findIntersection(scene.position[i], scene.radius[i], rayOrigin, rayDir);
		if (dist > 0 && dist < minDist) {
			minDist = dist;
			idx = i;
		}
	}

	if (idx >= 0)
	{
		vec3 p = rayOrigin + rayDir * minDist;
		vec3 col = getSphereColor(scene.position[idx], scene.color[idx], p);

		pixel[0] = col.x;
		pixel[1] = col.y;
		pixel[2] = col.z;
	}
	else {
		pixel[0] = 0.0;
		pixel[1] = 1.0f;
		pixel[2] = 0.0f;

	}
}

extern "C"
void cuda_texture_2d(Screen screen, Scene scene)
{
	cudaError_t error = cudaSuccess;

	//dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	//dim3 Dg = dim3((screen.width + Db.x - 1) / Db.x, (screen.width + Db.y - 1) / Db.y);
	dim3 Db = dim3(32, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3(screen.width / Db.x, screen.height / Db.y);

	cuda_kernel_texture_2d << <Dg, Db >> > (screen, scene);

	error = cudaGetLastError();

	if (error != cudaSuccess)
	{
		printf("cuda_kernel_texture_2d() failed to launch error = %d\n", error);
	}
}
