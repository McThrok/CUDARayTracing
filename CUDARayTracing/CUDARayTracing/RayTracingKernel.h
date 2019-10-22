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

#define NAME_LEN 512

class RayTracingKernel
{
public:
	cudaGraphicsResource* cudaResource;
	float* spheres;
	int spheres_num;
	int width;
	int height;
	size_t pitch;

	void* cudaLinearMemory;
	float* colors;

	bool findCUDADevice();

	void Run();
	void Cleanup();
	void InitSpheres();

	//CPU
	void InitCPU();
	void RunCPU();
	void CopyToGPU();
};

