#pragma once
#include "Sphere.h"
#include "Camera.h"

class Scene {
public:
	Sphere* spheres;
	int sphere_num;

	Camera cam;

};