/*modified from the orginal Titan simulation libaray:https://github.com/jacobaustin123/Titan
ref: J. Austin, R. Corrales-Fatou, S. Wyetzner, and H. Lipson, “Titan: A Parallel Asynchronous Library for Multi-Agent and Soft-Body Robotics using NVIDIA CUDA,” ICRA 2020, May 2020.
*/

#ifndef VEC_H
#define VEC_H

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#ifdef __CUDACC__
#define CUDA_DEVICE __device__
#else
#define CUDA_DEVICE
#endif


#if defined(__CUDACC__) // NVCC
#define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
#define MY_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
#define MY_ALIGN(n) __declspec(align(n))
#else
#error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif //https://stackoverflow.com/questions/12778949/cuda-memory-alignment/12779757

#include <iostream>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>

/*
index 2 (int)
*/
struct MY_ALIGN(8) Vec2i {
	int x, y;

	CUDA_CALLABLE_MEMBER Vec2i() {
		x = 0;
		y = 0;
	} // default
	CUDA_CALLABLE_MEMBER Vec2i(const Vec2i& v) {
		x = v.x;
		y = v.y;
	} // copy constructor
	CUDA_CALLABLE_MEMBER Vec2i(int x, int y) {
		this->x = x;
		this->y = y;
	} // initialization from x, y values

	CUDA_CALLABLE_MEMBER Vec2i& operator=(const Vec2i& v) {
		x = v.x;
		y = v.y;
		return *this;
	}
	Vec2i(const std::vector<int>& v) {
		x = v[0];
		y = v[1];
	}
	Vec2i& operator=(const std::vector<int>& v) {
		x = v[0];
		y = v[1];
		return *this;
	}
};

/*
vector 3 (double)
use align to force alignment for gpu memory
*/
struct MY_ALIGN(8) Vec3d {
	double x, y, z;

	CUDA_CALLABLE_MEMBER Vec3d() {
		x = 0;
		y = 0;
		z = 0;
	} // default

	CUDA_CALLABLE_MEMBER Vec3d(const Vec3d& v) {
		x = v.x;
		y = v.y;
		z = v.z;
	} // copy constructor
	//CUDA_CALLABLE_MEMBER Vec3d(const Vec3d& v) = default;

	CUDA_CALLABLE_MEMBER Vec3d(double x, double y, double z) {
		this->x = x;
		this->y = y;
		this->z = z;
	} // initialization from x, y, and z values

	CUDA_CALLABLE_MEMBER Vec3d& operator=(const Vec3d& v) {
		x = v.x;
		y = v.y;
		z = v.z;
		return *this;
	}

	Vec3d(const std::vector<double>& v) {
		x = v[0];
		y = v[1];
		z = v[2];
	}

	Vec3d& operator=(const std::vector<double>& v) {
		x = v[0];
		y = v[1];
		z = v[2];
		return *this;
	}

	inline CUDA_CALLABLE_MEMBER Vec3d& operator+=(const Vec3d& v) {
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}

	inline CUDA_CALLABLE_MEMBER Vec3d& operator-=(const Vec3d& v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}

	inline CUDA_CALLABLE_MEMBER Vec3d& operator*=(const Vec3d& v) {
		x *= v.x;
		y *= v.y;
		z *= v.z;
		return *this;
	}

	inline CUDA_CALLABLE_MEMBER Vec3d& operator*=(const double& d) {
		x *= d;
		y *= d;
		z *= d;
		return *this;
	}

	inline CUDA_CALLABLE_MEMBER Vec3d& operator/=(const Vec3d& v) {
		x /= v.x;
		y /= v.y;
		z /= v.z;
		return *this;
	}

	inline CUDA_CALLABLE_MEMBER Vec3d& operator/=(const double& d) {
		x /= d;
		y /= d;
		z /= d;
		return *this;
	}

	inline CUDA_CALLABLE_MEMBER Vec3d operator-() const {
		return Vec3d(-x, -y, -z);
	}

	CUDA_CALLABLE_MEMBER double& operator [] (int n) {
		switch (n){
		case 0:
			return x;
		case 1:
			return y;
		case 2:
			return z;
		default:
			fprintf(stderr, "C FILE %s LINE %d:operator [n] out of range!\n", __FILE__, __LINE__);
			exit(-1);
		} // to do remove this
	}

	CUDA_CALLABLE_MEMBER const double& operator [] (int n) const {
		switch (n){
		case 0:
			return x;
		case 1:
			return y;
		case 2:
			return z;
		default:
			fprintf(stderr, "C FILE %s LINE %d:operator [n] out of range!\n", __FILE__, __LINE__);
			exit(-1);
		} // to do remove this
	}
	inline CUDA_CALLABLE_MEMBER friend Vec3d operator+(const Vec3d& v1, const Vec3d& v2) {
		return Vec3d(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
	}
	inline CUDA_CALLABLE_MEMBER friend Vec3d operator-(const Vec3d& v1, const Vec3d& v2) {
		return Vec3d(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
	}
	inline CUDA_CALLABLE_MEMBER friend Vec3d operator*(const double x, const Vec3d& v) {
		return Vec3d(v.x * x, v.y * x, v.z * x);
	}
	inline CUDA_CALLABLE_MEMBER friend Vec3d operator*(const Vec3d& v, const double x) {
		return x * v;
	} // double times Vec3d
	inline CUDA_CALLABLE_MEMBER friend bool operator==(const Vec3d& v1, const Vec3d& v2) {
		return (v1[0] == v2[0] && v1[1] == v2[1] && v1[2] == v2[2]);
	}
	inline CUDA_CALLABLE_MEMBER friend Vec3d operator*(const Vec3d& v1, const Vec3d& v2) {
		return Vec3d(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
	} // Multiplies two Vecs (elementwise)
	inline CUDA_CALLABLE_MEMBER friend Vec3d operator/(const Vec3d& v, const double x) {
		return Vec3d(v.x / x, v.y / x, v.z / x);
	} //  vector over double
	inline CUDA_CALLABLE_MEMBER friend Vec3d operator/(const Vec3d& v1, const Vec3d& v2) {
		return Vec3d(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
	} // divides two Vecs (elementwise)

	friend std::ostream& operator << (std::ostream& strm, const Vec3d& v) {
		return strm << "(" << v[0] << ", " << v[1] << ", " << v[2] << ")";
	} // print

	CUDA_CALLABLE_MEMBER void print() {
		printf("(%2f, %2f, %2f)\n", x, y, z);
	}

	inline CUDA_CALLABLE_MEMBER  double SquaredSum() const {
		return x * x + y * y + z * z;
	}

	inline CUDA_CALLABLE_MEMBER  double norm() const {
#ifdef __CUDA_ARCH__ 
		return norm3d(x, y, z);
#else
		return sqrt(x*x + y*y + z*z);
#endif
	} // gives vector norm

	CUDA_CALLABLE_MEMBER Vec3d normalize() {
		double n = this->norm();
		//if (n<1e-8)
		//{// Todo: change this
		//    n = 1e-8;// add for numerical stability
		//}
		*this /= n;
		return *this;
	} // return the normalized vector

	inline CUDA_CALLABLE_MEMBER double sum() const {
#ifdef __CUDA_ARCH__
		return fma(x, y, z); // compute x+y+z as a single operation:https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1gff2117f6f3c4ff8a2aa4ce48a0ff2070
#else
		return x + y + z;
#endif
		
	} // sums all components of the vector


	inline CUDA_CALLABLE_MEMBER void setZero() {
		x = 0;
		y = 0;
		z = 0;
	}

	inline CUDA_CALLABLE_MEMBER double dot(const Vec3d& b) { // dot product
		return x * b.x + y * b.y + z * b.z; // preferably use this version
	}

	inline friend CUDA_CALLABLE_MEMBER double dot(const Vec3d& a, const Vec3d& b){
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}// dot product

	inline friend CUDA_CALLABLE_MEMBER Vec3d cross(const Vec3d& v1, const Vec3d& v2) {
		return Vec3d(v1.y * v2.z - v1.z * v2.y, v2.x * v1.z - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
	}

	inline CUDA_DEVICE void Vec3d::atomicVecAdd(const Vec3d& v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
		atomicAdd(&x, v.x);
		atomicAdd(&y, v.y);
		atomicAdd(&z, v.z);
#endif
	}

	// https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
	// rotate a vector {v_} with rotation axis {k} anchored at point {offset} by {theta} [rad]
	friend CUDA_CALLABLE_MEMBER Vec3d AxisAngleRotaion(const Vec3d& k, const Vec3d& v_, const double& theta, const Vec3d& offset);



	inline friend CUDA_CALLABLE_MEMBER double angleBetween(Vec3d p0, Vec3d p1) {
		return acos(p0.dot(p1) / (p0.norm() * p1.norm()));
	}

	friend CUDA_CALLABLE_MEMBER double signedAngleBetween(Vec3d p0, Vec3d p1, Vec3d normal);


	friend CUDA_CALLABLE_MEMBER Vec3d slerp(Vec3d p0, Vec3d p1, double t);
};





#endif
