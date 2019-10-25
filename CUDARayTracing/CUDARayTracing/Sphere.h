#pragma once

#include "math.h"
#include "Ray.h"
#include <DirectXMath.h>

using namespace DirectX;

class Sphere {
public:
	XMFLOAT3 position;
	float radius;
	XMFLOAT4 color;

	//Sphere();
	//Sphere(XMFLOAT3 position, float radius, XMFLOAT4 color);


	XMFLOAT3 getNormalAt(XMFLOAT3 point) {
		// normal always points away from the center of a sphere
		XMFLOAT3 normal;
		XMStoreFloat3(&normal, XMVector3Normalize(XMLoadFloat3(&point) - XMLoadFloat3(&position)));

		return normal;
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

	Sphere::Sphere() {
		position = XMFLOAT3(0, 0, 0);
		radius = 1.0;
		color = { 0.5,0.5,0.5, 0 };
	}

	Sphere::Sphere(XMFLOAT3 position, float radius, XMFLOAT4 color) {
		this->position = position;
		this->radius = radius;
		this->color = color;
	}
};

