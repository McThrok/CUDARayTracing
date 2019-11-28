#pragma once
#include "Scene.h"
#include "Sphere.h"
#include "Camera.h"

#include <dynlink_d3d11.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
#include <d3dcompiler.h>
#include <helper_cuda.h>
#include <helper_functions.h>   

class SceneManager
{
public:
	Scene scene;

	SceneManager() {}
	Camera* cam;

	void Cleanup();
	void UpdateCamera();

};

