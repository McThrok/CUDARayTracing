#pragma once
#include "Scene.h"
#include "Sphere.h"
#include "Plane.h"
#include "Camera.h"
#include "Light.h"

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

	Camera* h_cam;
	Light* h_light;

	void Cleanup() {
		checkCudaErrors(cudaFree(scene.spheres));
		checkCudaErrors(cudaFree(scene.planes));
		checkCudaErrors(cudaFree(scene.light));
		checkCudaErrors(cudaFree(scene.cam));

		delete h_cam;
		delete h_light;
	}

	void UpdateCamera() {
		checkCudaErrors(cudaMalloc((void**)&scene.cam, sizeof(Camera)));
		checkCudaErrors(cudaMemcpy(scene.cam, h_cam, sizeof(Camera), cudaMemcpyHostToDevice));
	}

	void UpdateLight() {
		checkCudaErrors(cudaMalloc((void**)&scene.light, sizeof(Light)));
		checkCudaErrors(cudaMemcpy(scene.light, h_light, sizeof(Light), cudaMemcpyHostToDevice));
	}

};

