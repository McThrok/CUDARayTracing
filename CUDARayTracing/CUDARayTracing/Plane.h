#pragma once

#include "math.h"
#include "vec3.h"
#include "Ray.h"
#include "Object.h"
#include "CudaCallableMember.h"

class Plane : public Object {
public:
	float width;
	float height;
	vec3 normal;

	CUDA Plane() : normal({ 0,1,0 }), width(0), height(0) {}

	CUDA Plane(vec3 _position, vec3 _normal, vec3 _color, float _width, float _height)
		: Object(_position, _color), normal(_normal), width(_width), height(_height) {}

	CUDA vec3 getNormalAt(vec3 point) override {
		return normal;
	}

	CUDA float findIntersection(Ray ray) override {
		float a = ray.direction.dot(-normal);

		if (a < 0) {
			return -1;
		}
		else {
			//NOT IMPLEMENTED !!!!
			return -1;
		}
	}

};