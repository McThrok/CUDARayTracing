#pragma once

#include "CudaCallableMember.h"
#include "Ray.h"
#include "vec3.h"

#define PI 3.1415926536f

class Camera
{
public:
	vec3 position;
	vec3 rotation;
	float movement_speed;
	float rotation_speed;

private:
	int width;
	int height;
	float aspect;
	float fov;

	bool recal_up = true;
	bool recal_fwd = true;
	vec3 forward;
	vec3 up;

public:
	CUDA Camera() {}

	CUDA Camera(int screenWidth, int screenHeight)
	{
		//LH or RH?
		rotation = { 0,0,0 };
		position = { 0,0,0 };
		movement_speed = 100;
		rotation_speed = 10;

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

		vec3 forward = GetForward();
		vec3 up = GetUp();
		vec3 right = forward.cross(up);
		vec3 dir = (forward + right * sinf(xAngle) * aspect + up * sinf(yAngle)).norm();

		return Ray(position, dir);
	}
	CUDA vec3 GetForward() {
		if (recal_fwd) {
			recal_fwd = false;
			forward = { -cosf(rotation.x) * sinf(rotation.y), sinf(rotation.x), -cosf(rotation.x) * cosf(rotation.y) };
		}
		return forward;
	}
	CUDA vec3 GetUp() {
		if (recal_up) {
			recal_up = false;
			up = {
				cosf(rotation.z) * sinf(rotation.x) * sinf(rotation.y) - cosf(rotation.y) * sinf(rotation.z),
				cosf(rotation.x) * cosf(rotation.z),
				cosf(rotation.y) * cosf(rotation.z) * sinf(rotation.x) + sinf(rotation.y) * sinf(rotation.z)
			};
		}
		return up;
	}
	CUDA vec3 GetRight() {
		return GetForward().cross(GetUp()).norm();
	}
	CUDA vec3 GetRotation() {
		return rotation;
	}
	CUDA void SetRotation(const vec3& r) {
		recal_fwd = true;
		recal_up = true;
		rotation = r;
	}

};

