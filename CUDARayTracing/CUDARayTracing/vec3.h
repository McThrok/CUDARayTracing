#pragma once

#include <math.h>

class vec3
{
public:
	float x, y, z;

	vec3() : vec3(0, 0, 0) {}
	vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
	vec3::vec3(const vec3& u);

	vec3& vec3::operator=(const vec3& u);
	vec3& vec3::operator+=(const vec3& u);
	vec3& vec3::operator-=(const vec3& u);
	vec3& vec3::operator*=(float t);
	vec3& vec3::operator*=(const vec3& u);
	vec3& vec3::operator/=(float t);

	const vec3 vec3::operator+(const vec3& u) const;
	const vec3 vec3::operator-(const vec3& u) const;
	const vec3 vec3::operator*(float t) const;
	const vec3 vec3::operator*(const vec3& u) const;
	const vec3 vec3::operator/(float t) const;

	vec3& vec3::operator-();
	bool vec3::operator==(const vec3& u) const;
	bool vec3::operator!=(const vec3& u) const;

	inline float vec3::length() const;
	inline float vec3::dot(const vec3& u) const;
	inline vec3 vec3::cross(const vec3& u) const;
};

