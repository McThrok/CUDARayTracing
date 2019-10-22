#pragma once

#include <DirectXMath.h>

using namespace DirectX;

class Camera
{
public:
	XMFLOAT3 position;
	XMFLOAT3 up;
	XMFLOAT3 forward;

private:
	XMMATRIX projection;

public:
	Camera(float aspect);
	XMMATRIX GetView();
	XMMATRIX GetProjection();
	XMMATRIX GetViewProjection();

	void SetAspect(float aspect);

private:
	void SetProjectionValues(float fovDegrees, float aspectRatio, float nearZ, float farZ);
};

