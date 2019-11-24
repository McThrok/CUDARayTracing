#include "SceneManager.h"


void SceneManager::Cleanup() {
	checkCudaErrors(cudaFree(scene.spheres));
	checkCudaErrors(cudaFree(scene.planes));
	checkCudaErrors(cudaFree(scene.light));
	checkCudaErrors(cudaFree(scene.cam));

	delete h_cam;
	delete h_light;
}

void SceneManager::UpdateCamera() {
	checkCudaErrors(cudaMalloc((void**)&scene.cam, sizeof(Camera)));
	checkCudaErrors(cudaMemcpy(scene.cam, h_cam, sizeof(Camera), cudaMemcpyHostToDevice));
}

void SceneManager::UpdateLight() {
	checkCudaErrors(cudaMalloc((void**)&scene.light, sizeof(Light)));
	checkCudaErrors(cudaMemcpy(scene.light, h_light, sizeof(Light), cudaMemcpyHostToDevice));
}