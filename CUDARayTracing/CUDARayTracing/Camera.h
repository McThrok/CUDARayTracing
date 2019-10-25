#pragma once

#include <DirectXMath.h>
#include <Ray.h>
#include "vec3.h"



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

