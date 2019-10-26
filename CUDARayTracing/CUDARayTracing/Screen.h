#pragma once

class Screen {
public:
	void* surface;
	int width;
	int height;
	size_t pitch;
};