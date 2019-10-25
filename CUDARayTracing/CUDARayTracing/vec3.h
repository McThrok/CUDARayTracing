#pragma once

#include <math.h>

class vec3
{
public:
	float x, y, z;

	vec3() : vec3(0, 0, 0) {}
	vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
	vec3(const vec3& u);

	vec3& operator=(const vec3& u);
	vec3& operator+=(const vec3& u);
	vec3& operator-=(const vec3& u);
	vec3& operator*=(float t);
	vec3& operator*=(const vec3& u);
	vec3& operator/=(float t);

	const vec3 operator+(const vec3& u) const;
	const vec3 operator-(const vec3& u) const;
	const vec3 operator*(float t) const;
	const vec3 operator*(const vec3& u) const;
	const vec3 operator/(float t) const;

	vec3& operator-();
	bool operator==(const vec3& u) const;
	bool operator!=(const vec3& u) const;

	float length() const;
	vec3 norm() const;
	float dot(const vec3& u) const;
	vec3 cross(const vec3& u) const;
};

