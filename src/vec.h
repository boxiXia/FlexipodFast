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

#include <iostream>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>

//#include<sys/types.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
//https://stackoverflow.com/questions/39274472/error-function-atomicadddouble-double-has-already-been-defined/39287554
static __inline__ __device__ double atomicDoubleAdd(double* address, double val) {
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	if (val == 0.0)
		return __longlong_as_double(old);
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}
#endif




// use align to force alignment for gpu memory
struct __align__(16) Vec {
	double x;
	double y;
	double z;


	//double data[3] = { 0 }; // initialize data to 0

	CUDA_CALLABLE_MEMBER Vec() {
		x = 0;
		y = 0;
		z = 0;
	} // default

	CUDA_CALLABLE_MEMBER Vec(const Vec& v) {
		x = v.x;
		y = v.y;
		z = v.z;
	} // copy constructor
	//CUDA_CALLABLE_MEMBER Vec(const Vec& v) = default;

	CUDA_CALLABLE_MEMBER Vec(double x, double y, double z) {
		this->x = x;
		this->y = y;
		this->z = z;
	} // initialization from x, y, and z values

	CUDA_CALLABLE_MEMBER Vec& operator=(const Vec& v) {
		x = v.x;
		y = v.y;
		z = v.z;
		return *this;
	}

	Vec(const std::vector<double>& v) {
		x = v[0];
		y = v[1];
		z = v[2];
	}

	Vec& operator=(const std::vector<double>& v) {
		x = v[0];
		y = v[1];
		z = v[2];
		return *this;
	}

	inline CUDA_CALLABLE_MEMBER Vec& operator+=(const Vec& v) {
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}

	inline CUDA_CALLABLE_MEMBER Vec& operator-=(const Vec& v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}

	inline CUDA_CALLABLE_MEMBER Vec& operator*=(const Vec& v) {
		x *= v.x;
		y *= v.y;
		z *= v.z;
		return *this;
	}

	inline CUDA_CALLABLE_MEMBER Vec& operator*=(const double& d) {
		x *= d;
		y *= d;
		z *= d;
		return *this;
	}

	inline CUDA_CALLABLE_MEMBER Vec& operator/=(const Vec& v) {
		x /= v.x;
		y /= v.y;
		z /= v.z;
		return *this;
	}

	inline CUDA_CALLABLE_MEMBER Vec& operator/=(const double& d) {
		x /= d;
		y /= d;
		z /= d;
		return *this;
	}

	inline CUDA_CALLABLE_MEMBER Vec operator-() const {
		return Vec(-x, -y, -z);
	}

	CUDA_CALLABLE_MEMBER double& operator [] (int n) {
		switch (n){
		case 0:
			return x;
		case 1:
			return y;
		case 2:
			return z;
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
		} // to do remove this
	}
	inline CUDA_CALLABLE_MEMBER friend Vec operator+(const Vec& v1, const Vec& v2) {
		return Vec(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
	}
	inline CUDA_CALLABLE_MEMBER friend Vec operator-(const Vec& v1, const Vec& v2) {
		return Vec(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
	}
	inline CUDA_CALLABLE_MEMBER friend Vec operator*(const double x, const Vec& v) {
		return Vec(v.x * x, v.y * x, v.z * x);
	}
	inline CUDA_CALLABLE_MEMBER friend Vec operator*(const Vec& v, const double x) {
		return x * v;
	} // double times Vec
	inline CUDA_CALLABLE_MEMBER friend bool operator==(const Vec& v1, const Vec& v2) {
		return (v1[0] == v2[0] && v1[1] == v2[1] && v1[2] == v2[2]);
	}
	inline CUDA_CALLABLE_MEMBER friend Vec operator*(const Vec& v1, const Vec& v2) {
		return Vec(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
	} // Multiplies two Vecs (elementwise)
	inline CUDA_CALLABLE_MEMBER friend Vec operator/(const Vec& v, const double x) {
		return Vec(v.x / x, v.y / x, v.z / x);
	} //  vector over double
	inline CUDA_CALLABLE_MEMBER friend Vec operator/(const Vec& v1, const Vec& v2) {
		return Vec(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
	} // divides two Vecs (elementwise)

	friend std::ostream& operator << (std::ostream& strm, const Vec& v) {
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

	CUDA_CALLABLE_MEMBER Vec normalize() {
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

	inline CUDA_CALLABLE_MEMBER double dot(const Vec& b) { // dot product
		return x * b.x + y * b.y + z * b.z; // preferably use this version
	}

	inline friend CUDA_CALLABLE_MEMBER double dot(const Vec& a, const Vec& b){
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}// dot product

	inline friend CUDA_CALLABLE_MEMBER Vec cross(const Vec& v1, const Vec& v2) {
		return Vec(v1.y * v2.z - v1.z * v2.y, v2.x * v1.z - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
	}

	//inline CUDA_DEVICE void atomicVecAdd(const Vec& v);

	inline CUDA_DEVICE void Vec::atomicVecAdd(const Vec& v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
		atomicAdd(&x, v.x);
		atomicAdd(&y, v.y);
		atomicAdd(&z, v.z);
#elif defined(__CUDA_ARCH__) &&__CUDA_ARCH__ < 600
		atomicDoubleAdd(&x, v.x);
		atomicDoubleAdd(&y, v.y);
		atomicDoubleAdd(&z, v.z);
#endif
	}

	// https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
	// rotate a vector {v_} with rotation axis {k} anchored at point {offset} by {theta} [rad]
	friend CUDA_CALLABLE_MEMBER Vec AxisAngleRotaion(const Vec& k, const Vec& v_, const double& theta, const Vec& offset);



	inline friend CUDA_CALLABLE_MEMBER double angleBetween(Vec p0, Vec p1) {
		return acos(p0.dot(p1) / (p0.norm() * p1.norm()));
	}

	friend CUDA_CALLABLE_MEMBER double signedAngleBetween(Vec p0, Vec p1, Vec normal);


	friend CUDA_CALLABLE_MEMBER Vec slerp(Vec p0, Vec p1, double t);
};





#endif
