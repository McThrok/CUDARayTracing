#pragma once
#include "vec3.h"
#include "CudaCallableMember.h"
#include "math.h"
#include "Camera.h"

class Light {
public:
	vec3 position;
	vec3 color;

	float ambient = 0.1f;
	float diffuse = 0.8f;
	float specular = 0.3f;
	int specular_m = 32;

	CUDA Light() : position({ 0,0,0 }), color({ 1,1,1 }) {}
	CUDA Light(vec3 p, vec3 c) : position(p), color(c) {}
	CUDA vec3 getColor(const Camera& cam, vec3 p, vec3 n, vec3 c) {

		vec3 toLight = (position - p).norm();
		vec3 toCam = (cam.position - p).norm();
		vec3 r = toLight.reflect(n.norm());

		float diff = toLight.dot(n);
		diff = diff > 0 ? diff : 0;

		float spec = r.dot(toCam);
		spec = spec > 0 ? spec : 0;

		return c * color * (diff * diffuse + specular * powf(spec, specular_m) + ambient);
	}

};

