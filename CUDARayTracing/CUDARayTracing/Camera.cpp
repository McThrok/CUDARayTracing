#include "Camera.h"

Camera::Camera(float aspect)
{
	//LH or RH?
	up = { 0,1,0 };
	forward = { 0,0,-1 };
	position = { 0,0,0 };

	SetAspect(aspect);
}

XMMATRIX Camera::GetView()
{
	//LH or RH?
	return XMMatrixLookToLH(XMLoadFloat3(&position), XMLoadFloat3(&forward), XMLoadFloat3(&up));
}

XMMATRIX Camera::GetProjection()
{
	return projection;
}

XMMATRIX Camera::GetViewProjection()
{
	return GetView() * GetProjection();
}

void Camera::SetAspect(float aspect)
{
	SetProjectionValues(90.0f, aspect, 0.1f, 3000.0f);
}

void Camera::SetProjectionValues(float fovDegrees, float aspectRatio, float nearZ, float farZ)
{
	float fovRadians = (fovDegrees / 360.0f) * XM_2PI;
	this->projection = XMMatrixPerspectiveFovLH(fovRadians, aspectRatio, nearZ, farZ);
}