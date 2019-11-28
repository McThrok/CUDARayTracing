#pragma once
#include "Scene.h"
#include "Camera.h"

#include <dynlink_d3d11.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
#include <d3dcompiler.h>
#include <helper_cuda.h>
#include <helper_functions.h>   
#include <random>


class SceneManager
{
public:
	Scene scene;

	Camera* cam;

	//mt19937 gen{ random_device{}() };
	std::mt19937 gen{ 0 };

	void InitScene(int screenWidth, int screenHeight);
	float getRandomFloat(float min, float max);
	void getRandomSphere(vec3& position, float& radius, vec3& color);

	void UpdateCamera();
	void Cleanup();

};

