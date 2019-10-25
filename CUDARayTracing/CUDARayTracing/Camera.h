#pragma once

#include <DirectXMath.h>
#include <Ray.h>
#include "vec3.h"

/*#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif*/ 

using namespace DirectX;

class Camera
{
public:
	vec3 position;
	vec3 up;
	vec3 forward;

private:
	int width;
	int height;
	float aspect;
	float fov;

public:
	Camera(int screenWidth, int screenHeight);
	Ray CastScreenRay(int x, int y);
};

