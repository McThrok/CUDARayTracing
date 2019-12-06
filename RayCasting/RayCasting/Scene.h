#pragma once
#include "Camera.h"

class Scene {
public:
	int sphere_num;

	vec3* position;
	float* radius;
	vec3* color;

	CameraData cam;
};