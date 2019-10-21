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

	bool findCUDADevice()
	{
		int nGraphicsGPU = 0;
		int deviceCount = 0;
		bool bFoundGraphics = false;
		char firstGraphicsName[NAME_LEN], devname[NAME_LEN];

		// This function call returns 0 if there are no CUDA capable devices.
		cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

		if (error_id != cudaSuccess)
		{
			printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
			exit(EXIT_FAILURE);
		}

		if (deviceCount == 0)
		{
			printf("> There are no device(s) supporting CUDA\n");
			return false;
		}
		else
		{
			printf("> Found %d CUDA Capable Device(s)\n", deviceCount);
		}

		// Get CUDA device properties
		cudaDeviceProp deviceProp;

		for (int dev = 0; dev < deviceCount; ++dev)
		{
			cudaGetDeviceProperties(&deviceProp, dev);
			STRCPY(devname, NAME_LEN, deviceProp.name);
			printf("> GPU %d: %s\n", dev, devname);
		}

		return true;
	}

	void Run();
	void Cleanup();
	void InitSpheres();
};

