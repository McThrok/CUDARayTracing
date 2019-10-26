#pragma once
#include "Sphere.h"
#include "Plane.h"
#include "Camera.h"
#include "Light.h"

class Scene {
public:
	Sphere* spheres;
	int sphere_num;

	Plane* planes;
	int plane_num;

	Light light;
	Camera cam;

};