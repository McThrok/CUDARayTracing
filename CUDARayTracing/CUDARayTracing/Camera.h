#pragma once

#include "CudaCallableMember.h"
#include "Ray.h"
#include "vec3.h"

#define PI 3.1415926536f

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
	CUDA Camera(int screenWidth, int screenHeight)
	{
		//LH or RH?
		up = { 0,1,0 };
		forward = { 0,0,-1 };
		position = { 0,0,0 };

		width = screenWidth;
		height = screenHeight;
		aspect = 1.0 * width / height;
		fov = (90.f / 360.0f) * PI;
	}

	CUDA Ray CastScreenRay(int x, int y)
	{
		//r=1
		float xAngle = fov * (1.0f * x / width - 0.5f);
		float yAngle = fov * (1.0f * y / height - 0.5f);

		vec3 right = forward.cross(up);
		vec3 dir = (forward + right * sinf(xAngle) * aspect + up * sinf(yAngle)).norm();

		return Ray(position, dir);
	}
};

