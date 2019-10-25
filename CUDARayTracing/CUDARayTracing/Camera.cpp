#include "Camera.h"

Camera::Camera(int screenWidth, int screenHeight)
{
	//LH or RH?
	up = { 0,1,0 };
	forward = { 0,0,-1 };
	position = { 0,0,0 };

	width = screenWidth;
	height = screenHeight;
	aspect = 1.0 * width / height;
	fov = (90.f / 360.0f) * XM_2PI;
}

Ray Camera::CastScreenRay(int x, int y)
{
	//r=1
	float xAngle = fov * (1.0f * x / width - 0.5f);
	float yAngle = fov * (1.0f * y / height - 0.5f);

	vec3 right = forward.cross(up);
	vec3 dir= (forward + right * sinf(xAngle) * aspect + up * sinf(yAngle)).norm();

	return Ray(position, dir);
}
