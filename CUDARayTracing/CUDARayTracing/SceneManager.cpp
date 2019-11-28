#include "SceneManager.h"


void SceneManager::Cleanup() {
	checkCudaErrors(cudaFree(scene.spheres));
	delete cam;
}

void SceneManager::UpdateCamera() {
	scene.cam = cam->GetData();
}

