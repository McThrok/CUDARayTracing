#pragma once

#include "vec3.h"
#include "Ray.h"
#include "CudaCallableMember.h"

class Object {
public:

	vec3 position;
	vec3 color;

	CUDA Object() :position({ 0,0,0 }), color({ 0.5f,0.5f,0.5f }) {}

	CUDA Object(vec3 _position, vec3 _color)
		: position(_position), color(color) {}

	CUDA virtual float findIntersection(Ray ray) = 0;
	CUDA virtual vec3 getNormalAt(vec3 point) = 0;

};
