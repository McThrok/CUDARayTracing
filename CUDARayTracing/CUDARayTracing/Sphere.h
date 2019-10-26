#pragma once

#include "math.h"
#include "Ray.h"
#include "vec3.h"
#include "Object.h"
#include "CudaCallableMember.h"

class Sphere : public Object {
public:
	float radius;

	CUDA Sphere::Sphere() : radius(1.0f) {}

	CUDA Sphere::Sphere(vec3 _position, vec3 _color, float _radius)
		: Object(_position, _color), radius(_radius) {}

	CUDA vec3 getNormalAt(vec3 point)  override {
		return (point - position).norm();
	}

	CUDA float findIntersection(Ray ray) override {
		float b = (2 * (ray.origin.x - position.x) * ray.direction.x) + (2 * (ray.origin.y - position.y) * ray.direction.y) + (2 * (ray.origin.z - position.z) * ray.direction.z);
		float c = pow(ray.origin.x - position.x, 2) + pow(ray.origin.y - position.y, 2) + pow(ray.origin.z - position.z, 2) - (radius * radius);

		float dist = -1;
		float discriminant = b * b - 4 * c;
		if (discriminant >= 0) {
			dist = ((-b - sqrt(discriminant)) / 2) - 0.000001;
			if (dist < 0)
				dist = ((-b + sqrt(discriminant)) / 2) - 0.000001;
		}

		return dist;
	}

};

