#include "RayTracingKernel.h"

extern "C"
{
	void cuda_texture_2d(Screen screen, Scene scene);
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

	cudaArray* cuArray;
	cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource, 0, 0);
	getLastCudaError("cudaGraphicsSubResourceGetMappedArray (cuda_texture_2d) failed");

	static Timer t;
	static int i = 0;
	i++;
	t.Start();
	// kick off the kernel and send the staging buffer screen.surface as an argument to allow the kernel to write to it
	cuda_texture_2d(screen, sm.scene);
	getLastCudaError("cuda_texture_2d failed");
	cudaDeviceSynchronize();
	t.Stop();
	if (i % 10 == 0)
		printf("inner %f\n", t.GetMilisecondsElapsed()/i);


	// then we want to copy screen.surface to the D3D texture, via its mapped form : cudaArray
	cudaMemcpy2DToArray(
		cuArray, // dst array
		0, 0,    // offset
		screen.surface, screen.pitch,       // src
		screen.width * 4 * sizeof(float), screen.height, // extent
		cudaMemcpyDeviceToDevice); // kind
	getLastCudaError("cudaMemcpy2DToArray failed");

	//
	// unmap the resources
	//
	cudaGraphicsUnmapResources(1, &cudaResource, 0);
	getLastCudaError("cudaGraphicsUnmapResources(1) failed");
}

bool RayTracingKernel::Init(int width, int height)
{
	screen.width = width;
	screen.height = height;

	if (!findCUDADevice())
		return false;

	InitScene();

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

void RayTracingKernel::InitScene() {
	sm.scene.plane_num = 0;
	sm.scene.sphere_num = 10;

	Sphere* h_spheres = new Sphere[sm.scene.sphere_num];
	for (int i = 0; i < sm.scene.sphere_num; i++)
	{
		h_spheres[i] = getRandomSphere();
	}

	checkCudaErrors(cudaMalloc((void**)&sm.scene.spheres, sizeof(Sphere) * sm.scene.sphere_num));
	checkCudaErrors(cudaMemcpy(sm.scene.spheres, h_spheres, sizeof(Sphere) * sm.scene.sphere_num, cudaMemcpyHostToDevice));

	delete h_spheres;

	sm.h_cam = new Camera(screen.width, screen.height);
	sm.UpdateCamera();

	sm.h_light = new Light({ 0,5,0 }, { 1,1,1 });
	sm.UpdateLight();
}

float RayTracingKernel::getRandomFloat(float min, float max) {
	return  uniform_real_distribution<float>{min, max}(gen);
}

Sphere RayTracingKernel::getRandomSphere() {
	Sphere s;
	s.radius = getRandomFloat(0.3, 3);

	s.color.x = getRandomFloat(0, 1);
	s.color.y = getRandomFloat(0, 1);
	s.color.z = getRandomFloat(0, 1);

	s.position.x = getRandomFloat(-10, 10);
	s.position.y = getRandomFloat(-5, 5);
	s.position.z = getRandomFloat(-15, -5);

	return s;
}
