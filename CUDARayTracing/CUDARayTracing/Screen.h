#pragma once

class Screen {
public:
	unsigned char* surface;
	int width;
	int height;
	size_t pitch;
};