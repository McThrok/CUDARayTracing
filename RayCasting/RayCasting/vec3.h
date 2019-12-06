#pragma once

#include <math.h>
#include "CudaCallableMember.h"

class vec3
{
public:
	float x, y, z;

	CUDA vec3() : vec3(0, 0, 0) {}
	CUDA vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
	CUDA vec3(const vec3& u)
	{
		x = u.x;
		y = u.y;
		z = u.z;
	}

	CUDA vec3& operator=(const vec3& u)
	{
		if (this != &u)
		{
			x = u.x;
			y = u.y;
			z = u.z;
		}
		return *this;
	}


	CUDA vec3& operator+=(const vec3& u)
	{
		x += u.x;
		y += u.y;
		z += u.z;
		return *this;
	}

	CUDA vec3& operator-=(const vec3& u)
	{
		x -= u.x;
		y -= u.y;
		z -= u.z;
		return *this;
	}

	CUDA vec3& operator*=(const vec3& u)
	{
		x *= u.x;
		y *= u.y;
		z *= u.z;
		return *this;
	}

	CUDA vec3& operator/=(const vec3& u)
	{
		x /= u.x;
		y /= u.y;
		z /= u.z;
		return *this;
	}


	CUDA vec3& operator+=(float t)
	{
		x += t;
		y += t;
		z += t;
		return *this;
	}

	CUDA vec3& operator-=(float t)
	{
		x -= t;
		y -= t;
		z -= t;
		return *this;
	}

	CUDA vec3& operator*=(float t)
	{
		x *= t;
		y *= t;
		z *= t;
		return *this;
	}

	CUDA vec3& operator/=(float t)
	{
		return *this *= (1 / t);
	}


	CUDA const vec3 operator+(const vec3& u) const
	{
		return vec3(*this) += u;
	}

	CUDA const vec3 operator-(const vec3& u) const
	{
		return vec3(*this) -= u;
	}

	CUDA const vec3 operator*(const vec3& u) const
	{
		return vec3(*this) *= u;
	}

	CUDA const vec3 operator/(const vec3& u) const
	{
		return vec3(*this) /= u;
	}


	CUDA const vec3 operator+(float t) const
	{
		return vec3(*this) += t;
	}

	CUDA const vec3 operator-(float t) const
	{
		return vec3(*this) -= t;
	}

	CUDA const vec3 operator*(float t) const
	{
		return vec3(*this) *= t;
	}

	CUDA const vec3 operator/(float t) const
	{
		return vec3(*this) *= (1 / t);
	}


	CUDA vec3 operator-()
	{
		return vec3(*this) *= -1.0f;
	}

	CUDA bool operator==(const vec3& u) const
	{
		return (x == u.x) && (y == u.y) && (z == u.z);
	}

	CUDA bool operator!=(const vec3& u) const
	{
		return (x != u.x) || (y != u.y) || (z != u.z);
	}


	CUDA float dot(const vec3& u) const
	{
		return x * u.x + y * u.y + z * u.z;
	}

	CUDA vec3 cross(const vec3& u) const
	{
		return vec3(y * u.z - z * u.y, z * u.x - x * u.z, x * u.y - y * u.x);
	}

	CUDA vec3 reflect(const vec3& u) const
	{
		return u * (*this).dot(u) / u.length() / u.length() * 2 - *this;
	}

	CUDA vec3 norm() const
	{
		return vec3(*this) /= length();
	}

	CUDA float length() const
	{
		return sqrt(x * x + y * y + z * z);
	}
};

