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

#define DIFFUSE 0.5
#define AMBIENT 0.2


__device__ vec3 getSphereColor(vec3& position, vec3& color, vec3& point) {
	vec3 toLight;
	float diff;

	vec3 normal = point - position;
	vec3 c = vec3(AMBIENT, AMBIENT, AMBIENT);

	toLight = (vec3( 1, 0, -0.5)).norm();
	diff = fmaxf(toLight.dot(normal), 0);
	c += (diff * DIFFUSE);


	toLight = (vec3(-1, 0, -0.5)).norm();
	diff = fmaxf(toLight.dot(normal), 0);
	c += (diff * DIFFUSE);

	toLight = (vec3( 0, 1, -0.5)).norm();
	diff = fmaxf(toLight.dot(normal), 0);
	c += (diff * DIFFUSE);

	return color * c;
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

__global__ void draw(Screen screen, Scene scene)
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
		pixel[1] = 0.0f;
		pixel[2] = 0.0f;

	}
}

extern "C"
void execute_kernel(Screen screen, Scene scene)
{
	cudaError_t error = cudaSuccess;

	//dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Db = dim3(32, 16);
	dim3 Dg = dim3(screen.width / Db.x, screen.height / Db.y);

	draw << <Dg, Db >> > (screen, scene);

	error = cudaGetLastError();

	if (error != cudaSuccess)
	{
		printf("cuda_kernel_texture_2d() failed to launch error = %d\n", error);
	}
}
