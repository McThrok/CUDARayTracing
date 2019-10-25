#pragma once

#include <DirectXMath.h>

using namespace DirectX;

class Ray {
public:
	XMFLOAT3 origin;
	XMFLOAT3 direction;

	//Ray();
	//Ray(XMFLOAT3 origin, XMFLOAT3 direction) ;

	Ray::Ray() {
		origin = { 0,0,0 };
		direction = { 1,0,0 };
	}

	Ray::Ray(XMFLOAT3 o, XMFLOAT3 d) {
		origin = o;
		direction = d;
	}
};