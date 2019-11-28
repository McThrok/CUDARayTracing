#include "SceneManager.h"

void SceneManager::InitScene(int screenWidth, int screenHeight) {
	cam = new Camera(screenWidth, screenHeight);
	UpdateCamera();

	scene.sphere_num = 32;

	vec3* h_position = new vec3[scene.sphere_num];
	float* h_radius = new float[scene.sphere_num];
	vec3* h_color = new vec3[scene.sphere_num];

	for (int i = 0; i < scene.sphere_num; i++)
		getRandomSphere(h_position[i], h_radius[i], h_color[i]);

	checkCudaErrors(cudaMalloc((void**)&scene.position, sizeof(vec3) * scene.sphere_num));
	checkCudaErrors(cudaMemcpy(scene.position, h_position, sizeof(vec3) * scene.sphere_num, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&scene.radius, sizeof(float) * scene.sphere_num));
	checkCudaErrors(cudaMemcpy(scene.radius, h_radius, sizeof(float) * scene.sphere_num, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&scene.color, sizeof(vec3) * scene.sphere_num));
	checkCudaErrors(cudaMemcpy(scene.color, h_color, sizeof(vec3) * scene.sphere_num, cudaMemcpyHostToDevice));

	delete h_position;
	delete h_radius;
	delete h_color;
}

float SceneManager::getRandomFloat(float min, float max) {
	return  std::uniform_real_distribution<float>{min, max}(gen);
}

void SceneManager::getRandomSphere(vec3& position, float& radius, vec3& color) {
	radius = getRandomFloat(0.3, 3);

	color.x = getRandomFloat(0, 1);
	color.y = getRandomFloat(0, 1);
	color.z = getRandomFloat(0, 1);

	position.x = getRandomFloat(-10, 10);
	position.y = getRandomFloat(-5, 5);
	position.z = getRandomFloat(-15, -5);

}

void SceneManager::Cleanup() {
	checkCudaErrors(cudaFree(scene.position));
	checkCudaErrors(cudaFree(scene.radius));
	checkCudaErrors(cudaFree(scene.color));
	delete cam;
}

void SceneManager::UpdateCamera() {
	scene.cam = cam->GetData();
}

