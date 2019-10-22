#include "RayTracingKernel.h"

extern "C"
{
	bool cuda_texture_2d(void* surface, size_t width, size_t height, size_t pitch, float* spheres, int num_sphere);
	bool cuda_copy_colors(void* surface, size_t width, size_t height, size_t pitch, float* colors);
}

void RayTracingKernel::Run()
{
	//
	// map the resources we've registered so we can access them in Cuda
	// - it is most efficient to map and unmap all resources in a single call,
	//   and to have the map/unmap calls be the boundary between using the GPU
	//   for Direct3D and Cuda
	//

	cudaGraphicsMapResources(1, &cudaResource, 0);
	getLastCudaError("cudaGraphicsMapResources(1) failed");

	//
	// run kernels which will populate the contents of those textures
	//
	// populate the 2d texture
	{
		cudaArray* cuArray;
		cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource, 0, 0);
		getLastCudaError("cudaGraphicsSubResourceGetMappedArray (cuda_texture_2d) failed");

		// kick off the kernel and send the staging buffer cudaLinearMemory as an argument to allow the kernel to write to it
		cuda_texture_2d(cudaLinearMemory, width, height, pitch, spheres, spheres_num);
		getLastCudaError("cuda_texture_2d failed");

		// then we want to copy cudaLinearMemory to the D3D texture, via its mapped form : cudaArray
		cudaMemcpy2DToArray(
			cuArray, // dst array
			0, 0,    // offset
			cudaLinearMemory, pitch,       // src
			width * 4 * sizeof(float), height, // extent
			cudaMemcpyDeviceToDevice); // kind
		getLastCudaError("cudaMemcpy2DToArray failed");
	}

	//
	// unmap the resources
	//
	cudaGraphicsUnmapResources(1, &cudaResource, 0);
	getLastCudaError("cudaGraphicsUnmapResources(1) failed");
}

void RayTracingKernel::InitCPU() {
	colors = new float[width * height * 4];
}

void RayTracingKernel::RunCPU()
{
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			float* pixel = colors + sizeof(float) * (y + x * height);
			pixel[0] = 1.0f;// 0.0 * x / width;
			pixel[1] = 1.0f;// 0.0 * y / height; // green
			pixel[2] = 1.0f; // blue
			pixel[3] = 1.0f; // alpha
		}
	}

	CopyToGPU();
}

void RayTracingKernel::CopyToGPU()
{
	cudaGraphicsMapResources(1, &cudaResource, 0);
	getLastCudaError("cudaGraphicsMapResources(1) failed");

	cudaArray* cuArray;
	cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource, 0, 0);
	getLastCudaError("cudaGraphicsSubResourceGetMappedArray (cuda_texture_2d) failed");

	checkCudaErrors(cudaMemcpy(cudaLinearMemory, colors, width * height * 4 * sizeof(float), cudaMemcpyHostToDevice));

	// then we want to copy cudaLinearMemory to the D3D texture, via its mapped form : cudaArray
	cudaMemcpy2DToArray(
		cuArray, // dst array
		0, 0,    // offset
		cudaLinearMemory, pitch,       // src
		width * 4 * sizeof(float), height, // extent
		cudaMemcpyDeviceToDevice); // kind
	getLastCudaError("cudaMemcpy2DToArray failed");

	cudaGraphicsUnmapResources(1, &cudaResource, 0);
	getLastCudaError("cudaGraphicsUnmapResources(1) failed");
}

void RayTracingKernel::Cleanup()
{
	checkCudaErrors(cudaGraphicsUnregisterResource(cudaResource));
	checkCudaErrors(cudaFree(cudaLinearMemory));
	checkCudaErrors(cudaFree(spheres));
}

void RayTracingKernel::InitSpheres() {
	spheres_num = 1;

	unsigned int mem_size = sizeof(float) * 4 * spheres_num;
	float* h_spheres = (float*)malloc(mem_size);

	h_spheres[0] = 0;
	h_spheres[1] = 0;
	h_spheres[2] = 0;
	h_spheres[3] = 0.5f;

	checkCudaErrors(cudaMalloc((void**)&spheres, mem_size));
	checkCudaErrors(cudaMemcpy(spheres, h_spheres, mem_size, cudaMemcpyHostToDevice));

	free(h_spheres);
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