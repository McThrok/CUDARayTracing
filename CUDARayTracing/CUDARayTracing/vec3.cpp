#include "vec3.h"
#pragma once

vec3::vec3(const vec3& u)
{
	x = u.x;
	y = u.y;
	z = u.z;
}

vec3& vec3::operator=(const vec3& u)
{
	if (this != &u)
	{
		x = u.x;
		y = u.y;
		z = u.z;
	}
	return *this;
}

vec3& vec3::operator+=(const vec3& u)
{
	x += u.x;
	y += u.y;
	z += u.z;
	return *this;
}

vec3& vec3::operator-=(const vec3& u)
{
	x -= u.x;
	y -= u.y;
	z -= u.z;
	return *this;
}

vec3& vec3::operator*=(float t)
{
	x *= t;
	y *= t;
	z *= t;
	return *this;
}

vec3& vec3::operator*=(const vec3& u)
{
	x *= u.x;
	y *= u.y;
	z *= u.z;
	return *this;
}

vec3& vec3::operator/=(float t)
{
	return *this *= (1 / t);
}


const vec3 vec3::operator+(const vec3& u) const
{
	return vec3(*this) += u;
}

const vec3 vec3::operator-(const vec3& u) const
{
	return vec3(*this) -= u;
}

const vec3 vec3::operator*(float t) const
{
	return vec3(*this) *= t;
}

const vec3 vec3::operator*(const vec3& u) const
{
	return vec3(*this) *= u;
}

const vec3 vec3::operator/(float t) const
{
	return vec3(*this) *= (1 / t);
}


vec3& vec3::operator-()
{
	return (*this) *= -1.0f;
}

bool vec3::operator==(const vec3& u) const
{
	return (x == u.x) && (y == u.y) && (z == u.z);
}

bool vec3::operator!=(const vec3& u) const
{
	return (x != u.x) || (y != u.y) || (z != u.z);
}


inline float vec3::dot(const vec3& u) const
{
	return x * u.x + y * u.y + z * u.z;
}

inline vec3 vec3::cross(const vec3& u) const
{
	return vec3(y * u.z - z * u.y, z * u.x - x * u.z, x * u.y - y * u.x;
}

inline float vec3::length() const
{
	return sqrt(x * x + y * y + z * z);
}

