#include "RayTracingKernel.h"

extern "C"
{
	bool cuda_texture_2d(void* surface, size_t width, size_t height, size_t pitch, float* spheres, int num_sphere);
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

void RayTracingKernel::Cleanup()
{
	checkCudaErrors(cudaGraphicsUnregisterResource(cudaResource));
	checkCudaErrors(cudaFree(cudaLinearMemory));
	checkCudaErrors(cudaFree(spheres));
}
