#pragma once
#include "vec3.h"
#include <DirectXMath.h>

using namespace DirectX;

class Ray {
public:
	vec3 origin;
	vec3 direction;

	Ray() :origin({ 0,0,0 }), direction({ 0,0,0 }) {	}
	Ray(vec3 o, vec3 d) :origin(o), direction(d) {	}
};