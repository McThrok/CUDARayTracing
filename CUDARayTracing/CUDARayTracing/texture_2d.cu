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

#define DIFFUSE 0.6
#define AMBIENT 0.2


__device__ vec3 getSphereColor(vec3& position, vec3& color, vec3& point) {
	vec3 lightPos = vec3(0, 5, 0);
	vec3 n = point - position;

	vec3 toLight = (lightPos - point).norm();
	float diff = fmaxf(toLight.dot(n), 0);
	return color * (diff * DIFFUSE + AMBIENT);
}

__device__ Ray castScreenRay(CameraData& c, int& x, int& y, int& width, int& height)
{
	//r=1
	float xAngle = c.fov * (1.0f * x / width - 0.5f);
	float yAngle = c.fov * (1.0f * y / height - 0.5f);

	vec3 dir = (c.forward + c.right * sinf(xAngle) * c.aspect + c.up * sinf(yAngle)).norm();

	return Ray(c.position, dir);
}

__device__ float findIntersection(vec3 & position, float & radius, Ray& ray) {
	float b = (2 * (ray.origin.x - position.x) * ray.direction.x) + (2 * (ray.origin.y - position.y) * ray.direction.y) + (2 * (ray.origin.z - position.z) * ray.direction.z);
	float c = pow(ray.origin.x - position.x, 2) + pow(ray.origin.y - position.y, 2) + pow(ray.origin.z - position.z, 2) - (radius * radius);

	float dist = -1;
	float discriminant = b * b - 4 * c;
	if (discriminant >= 0) {
		float tmp = sqrtf(discriminant);
		dist = ((-b - tmp) / 2) - 0.000001;
		if (dist < 0)
			dist = ((-b + tmp) / 2) - 0.000001;
	}

	return dist;


}
__global__ void cuda_kernel_texture_2dx(Screen screen, Scene scene)
{
	int si = threadIdx.x;
	int x = threadIdx.y + blockDim.x * blockIdx.x;
	int y = blockIdx.y;

	if (x >= screen.width || y >= screen.height) return;

	float* pixel = (float*)((char*)screen.surface + y * screen.pitch) + 4 * x;

	Ray ray = castScreenRay(scene.cam, x, y, screen.width, screen.height);

	pixel[3] = 1.0f; // alpha

	float minDist = 10000;
	int idx = -1;

	for (int i = 0; i < scene.sphere_num; i++)
	{
		float dist = findIntersection(scene.position[i], scene.radius[i], ray);
		if (dist > 0 && dist < minDist) {
			minDist = dist;
			idx = i;
		}
	}
	if (idx >= 0)
	{
		vec3 p = ray.getPointAt(minDist);
		vec3 col = getSphereColor(scene.position[idx], scene.color[idx], p);

		pixel[0] = col.x;// 0.0 * x / width;
		pixel[1] = col.y;// 0.0 * y / height; // green
		pixel[2] = col.z; // blue
	}
	else {
		pixel[0] = 0.0f;
		pixel[1] = 1.0f;
		pixel[2] = 0.0f; // blue

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

	Ray ray = castScreenRay(scene.cam, x, y, screen.width, screen.height);

	pixel[3] = 1.0f; // alpha

	float minDist = 10000;
	int idx = -1;

	for (int i = 0; i < scene.sphere_num; i++)
	{
		float dist = findIntersection(scene.position[i], scene.radius[i], ray);
		if (dist > 0 && dist < minDist) {
			minDist = dist;
			idx = i;
		}
	}
	if (idx >= 0)
	{
		vec3 p = ray.getPointAt(minDist);
		vec3 col = getSphereColor(scene.position[idx], scene.color[idx], p);

		pixel[0] = col.x;// 0.0 * x / width;
		pixel[1] = col.y;// 0.0 * y / height; // green
		pixel[2] = col.z; // blue
	}
	else {
		pixel[0] = 0.0f;
		pixel[1] = 1.0f;
		pixel[2] = 0.0f; // blue

	}
}

extern "C"
void cuda_texture_2dx(Screen screen, Scene scene)
{
	cudaError_t error = cudaSuccess;

	dim3 gridSize = dim3(screen.width / 32, screen.height, 1);
	//dim3 gridSize = dim3(screen.width / 32, 100, 1);
	dim3 blockSize = dim3(32, 32, 1);

	cuda_kernel_texture_2d << < gridSize, blockSize >> > (screen, scene);

	error = cudaGetLastError();

	if (error != cudaSuccess)
	{
		printf("cuda_kernel_texture_2d() failed to launch error = %d\n", error);
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
