#pragma once

#include <math.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

class vec3
{
public:
	float x, y, z;

	CUDA_CALLABLE_MEMBER vec3() : vec3(0, 0, 0) {}
	CUDA_CALLABLE_MEMBER ~vec3() {}
	CUDA_CALLABLE_MEMBER vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
	CUDA_CALLABLE_MEMBER vec3(const vec3& u)
	{
		x = u.x;
		y = u.y;
		z = u.z;
	}

	CUDA_CALLABLE_MEMBER vec3& operator=(const vec3& u)
	{
		if (this != &u)
		{
			x = u.x;
			y = u.y;
			z = u.z;
		}
		return *this;
	}


	CUDA_CALLABLE_MEMBER vec3& operator+=(const vec3& u)
	{
		x += u.x;
		y += u.y;
		z += u.z;
		return *this;
	}

	CUDA_CALLABLE_MEMBER vec3& operator-=(const vec3& u)
	{
		x -= u.x;
		y -= u.y;
		z -= u.z;
		return *this;
	}

	CUDA_CALLABLE_MEMBER vec3& operator*=(const vec3& u)
	{
		x *= u.x;
		y *= u.y;
		z *= u.z;
		return *this;
	}

	CUDA_CALLABLE_MEMBER vec3& operator/=(const vec3& u)
	{
		x /= u.x;
		y /= u.y;
		z /= u.z;
		return *this;
	}


	CUDA_CALLABLE_MEMBER vec3& operator+=(float t)
	{
		x += t;
		y += t;
		z += t;
		return *this;
	}

	CUDA_CALLABLE_MEMBER vec3& operator-=(float t)
	{
		x -= t;
		y -= t;
		z -= t;
		return *this;
	}

	CUDA_CALLABLE_MEMBER vec3& operator*=(float t)
	{
		x *= t;
		y *= t;
		z *= t;
		return *this;
	}

	CUDA_CALLABLE_MEMBER vec3& operator/=(float t)
	{
		return *this *= (1 / t);
	}


	CUDA_CALLABLE_MEMBER const vec3 operator+(const vec3& u) const
	{
		return vec3(*this) += u;
	}

	CUDA_CALLABLE_MEMBER const vec3 operator-(const vec3& u) const
	{
		return vec3(*this) -= u;
	}

	CUDA_CALLABLE_MEMBER const vec3 operator*(const vec3& u) const
	{
		return vec3(*this) *= u;
	}

	CUDA_CALLABLE_MEMBER const vec3 operator/(const vec3& u) const
	{
		return vec3(*this) /= u;
	}


	CUDA_CALLABLE_MEMBER const vec3 operator+(float t) const
	{
		return vec3(*this) += t;
	}

	CUDA_CALLABLE_MEMBER const vec3 operator-(float t) const
	{
		return vec3(*this) -= t;
	}

	CUDA_CALLABLE_MEMBER const vec3 operator*(float t) const
	{
		return vec3(*this) *= t;
	}

	CUDA_CALLABLE_MEMBER const vec3 operator/(float t) const
	{
		return vec3(*this) *= (1 / t);
	}


	CUDA_CALLABLE_MEMBER vec3& operator-()
	{
		return (*this) *= -1.0f;
	}

	CUDA_CALLABLE_MEMBER bool operator==(const vec3& u) const
	{
		return (x == u.x) && (y == u.y) && (z == u.z);
	}

	CUDA_CALLABLE_MEMBER bool operator!=(const vec3& u) const
	{
		return (x != u.x) || (y != u.y) || (z != u.z);
	}


	CUDA_CALLABLE_MEMBER float dot(const vec3& u) const
	{
		return x * u.x + y * u.y + z * u.z;
	}

	CUDA_CALLABLE_MEMBER vec3 cross(const vec3& u) const
	{
		return vec3(y * u.z - z * u.y, z * u.x - x * u.z, x * u.y - y * u.x);
	}

	CUDA_CALLABLE_MEMBER vec3 norm() const
	{
		return vec3(*this) /= length();
	}

	CUDA_CALLABLE_MEMBER float length() const
	{
		return sqrt(x * x + y * y + z * z);
	}
};

