#pragma once

#include "Ray.h"
#include "vec3.h"

#define PI 3.1415926536f

class CameraData
{
public:
	float aspect;
	float fov;

	vec3 position;
	vec3 forward;
	vec3 up;
	vec3 right;

};

class Camera
{
public:
	vec3 position;
	vec3 rotation;//z==0
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
	Camera() {}

	Camera(int screenWidth, int screenHeight)
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

	Ray CastScreenRay(int x, int y)
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
	vec3 GetForward() {
		if (recal_fwd) {
			forward = { -cosf(rotation.x) * sinf(rotation.y), sinf(rotation.x), -cosf(rotation.x) * cosf(rotation.y) };
			recal_fwd = false;
		}
		return forward;
	}
	vec3 GetUp() {
		if (recal_up) {
			up = {
				sinf(rotation.x) * sinf(rotation.y),
				cosf(rotation.x),
				cosf(rotation.y) * sinf(rotation.x)
			};
			recal_up = false;
		}
		return up;
	}
	vec3 GetRight() {
		return GetForward().cross(GetUp()).norm();
	}
	vec3 GetRotation() {
		return rotation;
	}
	void SetRotation(const vec3& r) {
		recal_fwd = true;
		recal_up = true;
		rotation = r;
	}

	CameraData GetData()
	{
		CameraData data;
		data.aspect = aspect;
		data.fov = fov;
		data.position = position;
		data.forward = GetForward();
		data.up = GetUp();
		data.right = GetRight();

		return data;
	}
};

