#pragma once
#include "vec3.h"
#include "CudaCallableMember.h"


class Ray {
public:
	vec3 origin;
	vec3 direction;

	CUDA Ray() :origin({ 0,0,0 }), direction({ 0,0,1 }) {}
	CUDA Ray(vec3 o, vec3 d) : origin(o), direction(d) {}

	CUDA vec3 getPointAt(float d) {
		return origin + direction * d;
	}
};