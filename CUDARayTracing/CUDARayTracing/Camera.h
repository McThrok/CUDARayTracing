#pragma once

#include <DirectXMath.h>
#include <Ray.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

using namespace DirectX;

class Camera
{
public:
	XMFLOAT3 position;
	XMFLOAT3 up;
	XMFLOAT3 forward;

private:
	XMMATRIX projection;
	int width;
	int height;
	float aspect;
	float fov;

public:
	CUDA_CALLABLE_MEMBER Camera(int screenWidth, int screenHeight);
	Ray CastScreenRay(int x, int y);
	CUDA_CALLABLE_MEMBER Ray CastScreenRay2(int x, int y);
};

