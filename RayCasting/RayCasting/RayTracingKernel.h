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
#include "Timer.h"

#include "Screen.h"
#include "SceneManager.h"
#include "vec3.h"

#define NAME_LEN 512

using namespace DirectX;
using namespace std;

class RayTracingKernel
{
public:
	cudaGraphicsResource* cudaResource;

	void* cudaLinearMemory;
	float* colors;

	SceneManager sm;
	Screen screen;


	void Run();
	bool Init(int width, int height);
	void RegisterTexture(ID3D11Texture2D* texture);
	void Cleanup();

private:
	bool findCUDADevice();
};

