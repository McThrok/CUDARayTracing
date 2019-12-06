#include "RayTracingKernel.h"

extern "C"
{
	void execute_kernel(Screen screen, Scene scene);
}

void RayTracingKernel::Run()
{
	cudaGraphicsMapResources(1, &cudaResource, 0);
	getLastCudaError("cudaGraphicsMapResources(1) failed");

	cudaArray* cuArray;
	cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource, 0, 0);
	getLastCudaError("cudaGraphicsSubResourceGetMappedArray (cuda_texture_2d) failed");

	execute_kernel(screen, sm.scene);
	cudaDeviceSynchronize();
	getLastCudaError("cuda_texture_2d failed");

	// then we want to copy screen.surface to the D3D texture, via its mapped form : cudaArray
	cudaMemcpy2DToArray(
		cuArray, // dst array
		0, 0,    // offset
		screen.surface, screen.pitch,       // src
		screen.width * 4 * sizeof(float), screen.height, // extent
		cudaMemcpyDeviceToDevice); // kind
	getLastCudaError("cudaMemcpy2DToArray failed");

	cudaGraphicsUnmapResources(1, &cudaResource, 0);
	getLastCudaError("cudaGraphicsUnmapResources(1) failed");
}

bool RayTracingKernel::Init(int width, int height)
{
	screen.width = width;
	screen.height = height;

	if (!findCUDADevice())
		return false;

	sm.InitScene(width, height);

	return true;
}

bool RayTracingKernel::findCUDADevice()
{
	int deviceCount = 0;
	char devname[NAME_LEN];

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

void RayTracingKernel::RegisterTexture(ID3D11Texture2D* texture)
{
	// 2D
	// register the Direct3D resources that we'll use
	// we'll read to and write from g_texture_2d, so don't set any special map flags for it
	checkCudaErrors(cudaGraphicsD3D11RegisterResource(&cudaResource, texture, cudaGraphicsRegisterFlagsNone));
	// cuda cannot write into the texture directly : the texture is seen as a cudaArray and can only be mapped as a texture
	// Create a buffer so that cuda can write into it
	// pixel fmt is DXGI_FORMAT_R32G32B32A32_FLOAT
	checkCudaErrors(cudaMallocPitch(&screen.surface, &screen.pitch, screen.width * sizeof(float) * 4, screen.height));
	cudaMemset(screen.surface, 1, screen.pitch * screen.height);
}

void RayTracingKernel::Cleanup()
{
	checkCudaErrors(cudaGraphicsUnregisterResource(cudaResource));
	checkCudaErrors(cudaFree(screen.surface));
	sm.Cleanup();
}
