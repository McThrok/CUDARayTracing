#pragma once

#include "math.h"
#include "vec3.h"
#include "Ray.h"
#include "Object.h"
#include "CudaCallableMember.h"

class Plane : public Object {
public:
	vec3 normal;

	CUDA Plane() : normal({ 0,1,0 }) {}

	CUDA Plane(vec3 _position, vec3 _normal, vec3 _color)
		: Object(_position, _color), normal(_normal) {}

	CUDA vec3 getNormalAt(vec3 point) override {
		return normal;
	}

	CUDA float findIntersection(Ray ray) override {
		float a = ray.direction.dot(-normal);

		if (a < 0) {
			return -1;
		}
		else {
			vec3 dist_vec = position - ray.origin;
			float dist = ray.direction.length() * dist_vec.dot(-normal) / ray.direction.dot(-normal);

			return dist;
		}
	}
};