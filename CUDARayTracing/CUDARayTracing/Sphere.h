#pragma once

#include "math.h"
#include "Ray.h"
#include "vec3.h"

#include <DirectXMath.h>

using namespace DirectX;

class Sphere {
public:
	vec3 position;
	float radius;
	vec3 color;

	Sphere::Sphere() :position({ 0,0,0 }), radius(1.0f), color({ 0.5,0.5,0.5 }) {}

	Sphere::Sphere(vec3 _position, float _radius, vec3 _color)
		: position(_position), radius(_radius), color(_color) {}


	vec3 getNormalAt(vec3 point) {
		// normal always points away from the center of a sphere
		return (point - position).norm();
	}

	float findIntersection(Ray ray) {
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

