#pragma once

#include <windows.h>
#include <mmsystem.h>

// This header inclues all the necessary D3D11 and CUDA includes
#include <dynlink_d3d11.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
#include <d3dcompiler.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

#include <DirectXMath.h>
#include "Camera.h"
#include "Sphere.h"
#include "Screen.h"
#include "Scene.h"

#define NAME_LEN 512

using namespace DirectX;

class RayTracingKernel
{
public:
	cudaGraphicsResource* cudaResource;
	int width;
	int height;
	bool cpu;
	size_t pitch;

	void* cudaLinearMemory;
	float* colors;

	Scene scene;
	//Screen screen;

	void Run();
	bool Init(int width, int height, bool cpu = false);
	void RegisterTexture(ID3D11Texture2D* texture);
	void Cleanup();

private:
	bool findCUDADevice();
	void InitScene();

	//GPU
	void RunGPU();

	//CPU
	void InitCPU();
	void RunCPU();
	void CopyToGPU();
};

