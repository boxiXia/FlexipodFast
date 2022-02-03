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
#define MY_ALIGN(data) __align__(data)
#elif defined(__GNUC__) // GCC
#define MY_ALIGN(data) __attribute__((aligned(data)))
#elif defined(_MSC_VER) // MSVC
#define MY_ALIGN(n) __declspec(align(n))
#else
#error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif //https://stackoverflow.com/questions/12778949/cuda-memory-alignment/12779757
// ref: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-vector-types

#include <iostream>
#include <cmath>
#include <vector>
#include <bitset>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>

#include <msgpack.hpp>

struct Vec2i; // index 2 (int)
struct Vec3i; // index 3 (int)
struct Vec3d; // vector 3 (double)
struct Mat3d; // matrix 3x3 (double)
class Vec8b; // boolean bit set


/*
index 2 (int)
*/
struct MY_ALIGN(8) Vec2i {
	int x, y;
//#ifndef __CUDACC__ // test false if they are currently being compiled by nvcc (device code).
	MSGPACK_DEFINE_ARRAY(x, y);
//#endif
	CUDA_CALLABLE_MEMBER Vec2i() {
		x = 0;
		y = 0;
	} // default
	CUDA_CALLABLE_MEMBER Vec2i(const Vec2i & v) {
		x = v.x;
		y = v.y;
	} // copy constructor
	CUDA_CALLABLE_MEMBER Vec2i(int x, int y) {
		this->x = x;
		this->y = y;
	} // initialization from x, y values
	CUDA_CALLABLE_MEMBER Vec2i& operator=(const Vec2i & v) {
		x = v.x;
		y = v.y;
		return *this;
	}
	Vec2i(const std::vector<int> & v) {
		x = v[0];
		y = v[1];
	}
	Vec2i& operator=(const std::vector<int> & v) {
		x = v[0];
		y = v[1];
		return *this;
	}
};

/*
index 3 (int)
*/
struct MY_ALIGN(4) Vec3i {
	int x, y, z;
//#ifndef __CUDACC__
	MSGPACK_DEFINE_ARRAY(x, y, z);
//#endif

	CUDA_CALLABLE_MEMBER Vec3i() {
		x = 0;
		y = 0;
		z = 0;
	} // default
	CUDA_CALLABLE_MEMBER Vec3i(const Vec3i & v) {
		x = v.x;
		y = v.y;
		z = v.z;
	} // copy constructor
	CUDA_CALLABLE_MEMBER Vec3i(int x, int y, int z) {
		this->x = x;
		this->y = y;
		this->z = z;
	} // initialization from x, y values
	CUDA_CALLABLE_MEMBER Vec3i& operator=(const Vec3i & v) {
		x = v.x;
		y = v.y;
		z = v.z;
		return *this;
	}
	Vec3i(const std::vector<int> &v) {
		x = v[0];
		y = v[1];
		z = v[2];
	}
	Vec3i& operator=(const std::vector<int> &v) {
		x = v[0];
		y = v[1];
		z = v[2];
		return *this;
	}
};

/*
vector 3 (double)
use align to force alignment for gpu memory
*/
struct MY_ALIGN(8) Vec3d {
	double x, y, z;
//#ifndef __CUDACC__
	MSGPACK_DEFINE_ARRAY(x, y, z);
//#endif

	CUDA_CALLABLE_MEMBER Vec3d() {
		x = 0;
		y = 0;
		z = 0;
	} // default

	CUDA_CALLABLE_MEMBER Vec3d(const Vec3d & v) {
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

	CUDA_CALLABLE_MEMBER Vec3d& operator=(const Vec3d & v) {
		x = v.x;
		y = v.y;
		z = v.z;
		return *this;
	}

	// fill a float array arr with the vector value
	// ref: https://stackoverflow.com/questions/5724171/passing-an-array-by-reference
	inline void fillArray(float(&arr)[3]) const {
		arr[0] = (float)x;
		arr[1] = (float)y;
		arr[2] = (float)z;
	}

	Vec3d(const std::vector<double> & v) {
		x = v[0];
		y = v[1];
		z = v[2];
	}

	Vec3d& operator=(const std::vector<double> & v) {
		x = v[0];
		y = v[1];
		z = v[2];
		return *this;
	}

	inline CUDA_CALLABLE_MEMBER Vec3d& operator+=(const Vec3d & v) {
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}

	inline CUDA_CALLABLE_MEMBER Vec3d& operator-=(const Vec3d & v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}

	inline CUDA_CALLABLE_MEMBER Vec3d& operator*=(const Vec3d & v) {
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

	inline CUDA_CALLABLE_MEMBER Vec3d& operator*=(const Vec8b& d);

	inline CUDA_CALLABLE_MEMBER Vec3d& operator/=(const Vec3d & v) {
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
		switch (n) {
		case 0:
			return x;
		case 1:
			return y;
		case 2:
			return z;
		default:
			printf("C FILE %s LINE %d:operator [n] out of range!\n", __FILE__, __LINE__);
			exit(-1);
		} // to do remove this
	}

	CUDA_CALLABLE_MEMBER const double& operator [] (int n) const {
		switch (n) {
		case 0:
			return x;
		case 1:
			return y;
		case 2:
			return z;
		default:
			printf("C FILE %s LINE %d:operator [n] out of range!\n", __FILE__, __LINE__);
			return x;
		} // to do remove this
	}

	//CUDA_CALLABLE_MEMBER friend double operator=(double* v1, const Vec3d& v2) {

	//}

	inline CUDA_CALLABLE_MEMBER friend Vec3d operator+(const Vec3d & v1, const Vec3d & v2) {
		return Vec3d(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
	}
	inline CUDA_CALLABLE_MEMBER friend Vec3d operator-(const Vec3d & v1, const Vec3d & v2) {
		return Vec3d(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
	}
	inline CUDA_CALLABLE_MEMBER friend Vec3d operator*(const double x, const Vec3d & v) {
		return Vec3d(v.x * x, v.y * x, v.z * x);
	}
	inline CUDA_CALLABLE_MEMBER friend Vec3d operator*(const Vec3d & v, const double x) {
		return x * v;
	} // double times Vec3d
	inline CUDA_CALLABLE_MEMBER friend bool operator==(const Vec3d & v1, const Vec3d & v2) {
		return (v1.x == v2.x && v1.y == v2.y && v1.z == v2.z);
	}
	inline CUDA_CALLABLE_MEMBER friend Vec3d operator*(const Vec3d & v1, const Vec3d & v2) {
		return Vec3d(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
	} // Multiplies two Vecs (elementwise)
	inline CUDA_CALLABLE_MEMBER friend Vec3d operator/(const Vec3d & v, const double x) {
		return Vec3d(v.x / x, v.y / x, v.z / x);
	} //  vector over double
	inline CUDA_CALLABLE_MEMBER friend Vec3d operator/(const Vec3d & v1, const Vec3d & v2) {
		return Vec3d(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
	} // divides two Vecs (elementwise)

	friend std::ostream& operator << (std::ostream & strm, const Vec3d & v) {
		return strm << "(" << v[0] << ", " << v[1] << ", " << v[2] << ")";
	} // print

	inline CUDA_CALLABLE_MEMBER void setZero() {
		x = 0;
		y = 0;
		z = 0;
	}

	CUDA_CALLABLE_MEMBER void print() {
		printf("(%2f, %2f, %2f)\n", x, y, z);
	}

	inline CUDA_CALLABLE_MEMBER double dot(const Vec3d& b) const { // dot product
		return x * b.x + y * b.y + z * b.z; // preferably use this version
	}

	inline friend CUDA_CALLABLE_MEMBER double dot(const Vec3d& a, const Vec3d& b) {
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}// dot product

	inline CUDA_CALLABLE_MEMBER  double SquaredSum() const {
		return x * x + y * y + z * z;
	}

	inline CUDA_CALLABLE_MEMBER double sum() const {
#ifdef __CUDA_ARCH__
		return fma(x, y, z); // compute x+y+z as a single operation:https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1gff2117f6f3c4ff8a2aa4ce48a0ff2070
#else
		return x + y + z;
#endif
	} // sums all components of the vector

	inline CUDA_CALLABLE_MEMBER  double norm() const {
#ifdef __CUDA_ARCH__ 
		return norm3d(x, y, z);
#else
		return sqrt(x * x + y * y + z * z);
#endif
	} // gives vector norm

	CUDA_CALLABLE_MEMBER Vec3d normalize() {
		double n = this->norm();
		//if (data<1e-8)
		//{// Todo: change this
		//    data = 1e-8;// add for numerical stability
		//}
		*this /= n;
		return *this;
	} // return the normalized vector


	// return a projection on to unit vector d
	inline CUDA_CALLABLE_MEMBER Vec3d project(const Vec3d & d) {
		return this->dot(d) * d;
	}

	// return a orthogonal decomposition of this with respect to unit vector d
	inline CUDA_CALLABLE_MEMBER Vec3d decompose(const Vec3d & d) {
		return *this - this->dot(d) * d;
	}

	inline CUDA_CALLABLE_MEMBER Vec3d cross(const Vec3d& v2) {
		return Vec3d(y * v2.z - z * v2.y, v2.x * z - x * v2.z, x * v2.y - y * v2.x);
	}

	inline friend CUDA_CALLABLE_MEMBER Vec3d cross(const Vec3d & v1, const Vec3d & v2) {
		return Vec3d(v1.y * v2.z - v1.z * v2.y, v2.x * v1.z - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
	}


	inline CUDA_DEVICE void atomicVecAdd(Vec3d& v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
		atomicAdd(&x, v.x);
		atomicAdd(&y, v.y);
		atomicAdd(&z, v.z);
#endif
	}

	// https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
	// rotate a vector {v_} with rotation axis {k} anchored at point {offset} by {theta} [rad]
	friend CUDA_CALLABLE_MEMBER Vec3d AxisAngleRotaion(const Vec3d & k, const Vec3d & v_, const double& theta, const Vec3d & offset);

	/* rotate a vector {v_} with rotation axis {axis_end-axis_start} by {theta} [rad] */
	friend CUDA_CALLABLE_MEMBER Vec3d AxisAngleRotaion(const Vec3d & axis_start, const Vec3d & axis_end, const Vec3d & v_, const double& theta);

	inline friend CUDA_CALLABLE_MEMBER double angleBetween(Vec3d p0, Vec3d p1) {
		return acos(p0.dot(p1) / (p0.norm() * p1.norm()));
	}

	friend CUDA_CALLABLE_MEMBER double signedAngleBetween(Vec3d p0, Vec3d p1, Vec3d normal);

	// linear interpolation
	friend CUDA_CALLABLE_MEMBER Vec3d lerp(Vec3d p0, Vec3d p1, double t);

	// spherical linear interpolation
	friend CUDA_CALLABLE_MEMBER Vec3d slerp(Vec3d p0, Vec3d p1, double t);
};

float lerp(float a, float b, float f);


struct MY_ALIGN(8) Mat3d {
	double m00, m01, m02, m10, m11, m12, m20, m21, m22;
//#ifndef __CUDACC__
	MSGPACK_DEFINE_ARRAY(m00, m01, m02, m10, m11, m12, m20, m21, m22);
//#endif


	// defualt constructor
	CUDA_CALLABLE_MEMBER Mat3d() {}

	CUDA_CALLABLE_MEMBER Mat3d(const Mat3d & o) {
		m00 = o.m00; m01 = o.m01; m02 = o.m02;
		m10 = o.m10; m11 = o.m11; m12 = o.m12;
		m20 = o.m20; m21 = o.m21; m22 = o.m22;
	} // copy constructor

	CUDA_CALLABLE_MEMBER Mat3d(
		const double& m00, const double& m01, const double& m02,
		const double& m10, const double& m11, const double& m12,
		const double& m20, const double& m21, const double& m22) {
		this->m00 = m00; this->m01 = m01; this->m02 = m02;
		this->m10 = m10; this->m11 = m11; this->m12 = m12;
		this->m20 = m20; this->m21 = m21; this->m22 = m22;
	} // initialization from values

	CUDA_CALLABLE_MEMBER Mat3d(const Vec3d & v0, const Vec3d & v1, const Vec3d & v2, bool row = true) {
		if (row) {// construct from row vector 
			m00 = v0.x; m01 = v0.y; m02 = v0.z;
			m10 = v1.x; m11 = v1.y; m12 = v1.z;
			m20 = v2.x; m21 = v2.y; m22 = v2.z;
		}
		else { // construct from column vector
			m00 = v0.x; m01 = v1.x; m02 = v2.x;
			m10 = v0.y; m11 = v1.y; m12 = v2.y;
			m20 = v0.z; m21 = v1.z; m22 = v2.z;
		}

	} // copy constructor fro Vector, rowwise

	CUDA_CALLABLE_MEMBER Mat3d& operator=(const Mat3d & o) {
		m00 = o.m00; m01 = o.m01; m02 = o.m02;
		m10 = o.m10; m11 = o.m11; m12 = o.m12;
		m20 = o.m20; m21 = o.m21; m22 = o.m22;
		return *this;
	}

	//inplaceelement-wise +
	inline CUDA_CALLABLE_MEMBER Mat3d& operator+=(const Mat3d & o) {
		m00 += o.m00; m01 += o.m01; m02 += o.m02;
		m10 += o.m10; m11 += o.m11; m12 += o.m12;
		m20 += o.m20; m21 += o.m21; m22 += o.m22;
		return *this;
	}
	//inplaceelement-wise +
	inline CUDA_CALLABLE_MEMBER Mat3d& operator+=(const double& d) {
		m00 += d; m01 += d; m02 += d;
		m10 += d; m11 += d; m12 += d;
		m20 += d; m21 += d; m22 += d;
		return *this;
	}

	//inplaceelement-wise -
	inline CUDA_CALLABLE_MEMBER Mat3d& operator-=(const Mat3d & o) {
		m00 -= o.m00; m01 -= o.m01; m02 -= o.m02;
		m10 -= o.m10; m11 -= o.m11; m12 -= o.m12;
		m20 -= o.m20; m21 -= o.m21; m22 -= o.m22;
		return *this;
	}
	//inplaceelement-wise -
	inline CUDA_CALLABLE_MEMBER Mat3d& operator-=(const double& d) {
		m00 -= d; m01 -= d; m02 -= d;
		m10 -= d; m11 -= d; m12 -= d;
		m20 -= d; m21 -= d; m22 -= d;
		return *this;
	}

	//inplaceelement-wise *
	inline CUDA_CALLABLE_MEMBER Mat3d& operator*=(const Mat3d & o) {
		m00 *= o.m00; m01 *= o.m01; m02 *= o.m02;
		m10 *= o.m10; m11 *= o.m11; m12 *= o.m12;
		m20 *= o.m20; m21 *= o.m21; m22 *= o.m22;
		return *this;
	}
	//inplaceelement-wise *
	inline CUDA_CALLABLE_MEMBER Mat3d& operator*=(const double& d) {
		m00 *= d; m01 *= d; m02 *= d;
		m10 *= d; m11 *= d; m12 *= d;
		m20 *= d; m21 *= d; m22 *= d;
		return *this;
	}

	//inplaceelement-wise /
	inline CUDA_CALLABLE_MEMBER Mat3d& operator/=(const Mat3d & o) {
		m00 /= o.m00; m01 /= o.m01; m02 /= o.m02;
		m10 /= o.m10; m11 /= o.m11; m12 /= o.m12;
		m20 /= o.m20; m21 /= o.m21; m22 /= o.m22;
		return *this;
	}
	//inplaceelement-wise /
	inline CUDA_CALLABLE_MEMBER Mat3d& operator/=(const double& d) {
		m00 /= d; m01 /= d; m02 /= d;
		m10 /= d; m11 /= d; m12 /= d;
		m20 /= d; m21 /= d; m22 /= d;
		return *this;
	}

	//element-wise negetive
	inline CUDA_CALLABLE_MEMBER Mat3d operator-() const {
		return Mat3d(-m00, -m01, -m02, -m10, -m11, -m12, -m20, -m21, -m22);
	}

	//element-wise +
	inline CUDA_CALLABLE_MEMBER friend Mat3d operator+(const Mat3d & a, const Mat3d & b) {
		return Mat3d(a) += b;
	}
	//element-wise +
	inline CUDA_CALLABLE_MEMBER friend Mat3d operator+(const Mat3d & a, const double& b) {
		return Mat3d(a) += b;
	}
	//element-wise +
	inline CUDA_CALLABLE_MEMBER friend Mat3d operator+(const double& a, const Mat3d & b) {
		return Mat3d(b) += a;
	}

	//element-wise -
	inline CUDA_CALLABLE_MEMBER friend Mat3d operator-(const Mat3d & a, const Mat3d & b) {
		return Mat3d(a) -= b;
	}
	//element-wise -
	inline CUDA_CALLABLE_MEMBER friend Mat3d operator-(const Mat3d & a, const double& b) {
		return Mat3d(a) -= b;
	}
	//element-wise -
	inline CUDA_CALLABLE_MEMBER friend Mat3d operator-(const double& a, const Mat3d & b) {
		return -b + a;
	}

	//element-wise *
	inline CUDA_CALLABLE_MEMBER friend Mat3d operator*(const Mat3d & a, const Mat3d & b) {
		return Mat3d(a) *= b;
	}
	//element-wise *
	inline CUDA_CALLABLE_MEMBER friend Mat3d operator*(const Mat3d & a, const double& b) {
		return Mat3d(a) *= b;
	}
	//element-wise *
	inline CUDA_CALLABLE_MEMBER friend Mat3d operator*(const double& a, const Mat3d & b) {
		return Mat3d(b) *= a;
	}

	//element-wise /
	inline CUDA_CALLABLE_MEMBER friend Mat3d operator/(const Mat3d & a, const Mat3d & b) {
		return Mat3d(a) /= b;
	}
	//element-wise /
	inline CUDA_CALLABLE_MEMBER friend Mat3d operator/(const Mat3d & a, const double& b) {
		return Mat3d(a) /= b;
	}
	//element-wise /
	inline CUDA_CALLABLE_MEMBER friend Mat3d operator/(const double& a, const Mat3d & b) {
		return Mat3d(a / b.m00, a / b.m01, a / b.m02, a / b.m10, a / b.m11, a / b.m12, a / b.m20, a / b.m21, a / b.m22);
	}

	// equlity 
	inline CUDA_CALLABLE_MEMBER friend bool operator==(const Mat3d & a, const Mat3d & b) {
		return	(a.m00 == b.m00) && (a.m01 == b.m01) && (a.m02 == b.m02) &&
			(a.m10 == b.m10) && (a.m11 == b.m11) && (a.m12 == b.m12) &&
			(a.m20 == b.m20) && (a.m21 == b.m21) && (a.m22 == b.m22);
	}

	friend std::ostream& operator << (std::ostream & strm, const Mat3d & a) {
		return strm << "(" << a.m00 << ", " << a.m01 << ", " << a.m02 << ")" << "\n"
			<< "(" << a.m10 << ", " << a.m11 << ", " << a.m12 << ")" << "\n"
			<< "(" << a.m20 << ", " << a.m21 << ", " << a.m22 << ")" << "\n";
	} // print

	CUDA_CALLABLE_MEMBER  void print() const {
		printf("(%2f, %2f, %2f)\n", m00, m01, m02);
		printf("(%2f, %2f, %2f)\n", m10, m11, m12);
		printf("(%2f, %2f, %2f)\n", m20, m21, m22);
	}

	// matrix transpose
	inline CUDA_CALLABLE_MEMBER  Mat3d transpose() const {
		return Mat3d(
			m00, m10, m20,
			m01, m11, m21,
			m02, m12, m22);
	}

	inline CUDA_CALLABLE_MEMBER Mat3d dot(const Mat3d & b) const { // matrix do product (multiplication)
		return Mat3d(
			m00 * b.m00 + m01 * b.m10 + m02 * b.m20, m00 * b.m01 + m01 * b.m11 + m02 * b.m21, m00 * b.m02 + m01 * b.m12 + m02 * b.m22,
			m10 * b.m00 + m11 * b.m10 + m12 * b.m20, m10 * b.m01 + m11 * b.m11 + m12 * b.m21, m10 * b.m02 + m11 * b.m12 + m12 * b.m22,
			m20 * b.m00 + m21 * b.m10 + m22 * b.m20, m20 * b.m01 + m21 * b.m11 + m22 * b.m21, m20 * b.m02 + m21 * b.m12 + m22 * b.m22
		);
		// preferably use this version
	}

	inline CUDA_CALLABLE_MEMBER Vec3d dot(const Vec3d & v) const { // matrix do product with a vector
		return Vec3d(m00 * v.x + m01 * v.y + m02 * v.z, m10 * v.x + m11 * v.y + m12 * v.z, m20 * v.x + m21 * v.y + m22 * v.z);
		// preferably use this version
	}

	inline friend CUDA_CALLABLE_MEMBER Mat3d dot(const Mat3d & a, const Mat3d & b) { // matrix do product (multiplication)
		return Mat3d(
			a.m00 * b.m00 + a.m01 * b.m10 + a.m02 * b.m20, a.m00 * b.m01 + a.m01 * b.m11 + a.m02 * b.m21, a.m00 * b.m02 + a.m01 * b.m12 + a.m02 * b.m22,
			a.m10 * b.m00 + a.m11 * b.m10 + a.m12 * b.m20, a.m10 * b.m01 + a.m11 * b.m11 + a.m12 * b.m21, a.m10 * b.m02 + a.m11 * b.m12 + a.m12 * b.m22,
			a.m20 * b.m00 + a.m21 * b.m10 + a.m22 * b.m20, a.m20 * b.m01 + a.m21 * b.m11 + a.m22 * b.m21, a.m20 * b.m02 + a.m21 * b.m12 + a.m22 * b.m22
		);
	}

	inline friend CUDA_CALLABLE_MEMBER Vec3d dot(const Mat3d & a, const Vec3d & v) { // matrix do product with a vector
		return Vec3d(a.m00 * v.x + a.m01 * v.y + a.m02 * v.z, a.m10 * v.x + a.m11 * v.y + a.m12 * v.z, a.m20 * v.x + a.m21 * v.y + a.m22 * v.z);
	}

	inline CUDA_CALLABLE_MEMBER double det() const { // matrix determinant
		return m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20) + m02 * (m10 * m21 - m11 * m20);
	}

	CUDA_CALLABLE_MEMBER Mat3d inv() const { // matrix inverse, must check det()!=0 before using this function
		double d = det();
		return (1. / d) * Mat3d(
			(m11 * m22 - m12 * m21), -(m01 * m22 - m02 * m21), (m01 * m12 - m02 * m11),
			-(m10 * m22 - m12 * m20), (m00 * m22 - m02 * m20), -(m00 * m12 - m02 * m10),
			(m10 * m21 - m11 * m20), -(m00 * m21 - m01 * m20), (m00 * m11 - m01 * m10));
	}


	inline CUDA_CALLABLE_MEMBER double trace() const { // matrix determinant
		return m00 + m11 + m22;
		// preferably use this version
	}

	inline CUDA_CALLABLE_MEMBER Vec3d const getRow(const int& k) const { // get row by index
		switch (k)
		{
		case 0:
			return Vec3d(m00, m01, m02);
		case 1:
			return Vec3d(m10, m11, m12);
		case 2:
			return Vec3d(m20, m21, m22);
		default:
			printf("C FILE %s LINE %d: row index out of range!\n", __FILE__, __LINE__);
			exit(-1);
		}
		// preferably use this version
	}

	inline CUDA_CALLABLE_MEMBER Vec3d getColumn(const int& k) const { // get column by index
		switch (k)
		{
		case 0:
			return Vec3d(m00, m10, m20);
		case 1:
			return Vec3d(m01, m11, m21);
		case 2:
			return Vec3d(m02, m12, m22);
		default:
			printf("C FILE %s LINE %d: column index out of range!\n", __FILE__, __LINE__);
			exit(-1);
		}
		// preferably use this version
	}

	static CUDA_CALLABLE_MEMBER Mat3d identity() {
		return Mat3d(1., 0., 0., 0., 1., 0., 0., 0., 1.);
	}

	//convert a rotation vector to rotation matrix
	static CUDA_CALLABLE_MEMBER Mat3d fromRotVec(Vec3d v) {
		// ref: https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
		double t = v.norm(); // angle theta [rad]
		double x = v.x / t;
		double y = v.y / t;
		double z = v.z / t;
		double c = cos(t);
		double s = sin(t);
		double a = 1 - c;
		return Mat3d(
			c + x * x * a, x * y * a - z * s, x * z * a + y * s,
			y * x * a + z * s, c + y * y * a, y * z * a - x * s,
			z * x * a - y * s, z * y * a + x * s, c + z * z * a
		);
	}

	/// <summary>
	/// Estimate world space angular velocity from two rotation matrix
	/// assuming uniform rotaiton and small dt
	/// </summary>
	/// <param name="r0">:initial rotation matrix</param>
	/// <param name="r1">:final rotation matrix</param>
	/// <param name="dt">:delta time [s] between inital and final rotation</param>
	/// <param name="body_space">:boolean, whether to return angular velocity in body space</param>
	/// <returns>
	/// angular velocity [rad/s] in body space or in world space 
	/// </returns>
	static CUDA_CALLABLE_MEMBER Vec3d angularVelocityFromRotation(Mat3d & r0, Mat3d & r1, const double& dt, bool body_space = false) {
		/// ref: https://math.stackexchange.com/questions/668866/how-do-you-find-angular-velocity-given-a-pair-of-3x3-rotation-matrices
		Mat3d a = r1.dot(r0.transpose());

		double c = (a.trace() - 1.) / 2.;
		if (c < -1.0 || c>1.0) { return Vec3d(0, 0, 0); }
		//c = (c < -1.0) ? -1.0 : (c > 1.0) ? 1.0 : c; // make sure cosine is in range [-1,1]
		double theta = acos(c);

		//Mat3d w = 0.5 / dt * theta / sin(theta) * (a - a.transpose());// skew symetric angular velocity matrix
		Mat3d w = 0.5 / dt * (abs(theta) < 1e-8 ? 1.0 : theta / sin(theta)) * (a - a.transpose());// skew symetric angular velocity matrix
		Vec3d av(w.m21, w.m02, w.m10); // angular velocity
		return body_space ? r1.transpose().dot(av) : av; // transform av to body space if body_space==true
	}

	friend void assertClose(const Mat3d & a, const Mat3d & b, double eps = 1e-15) {
		assert(abs((a - b).det()) < eps);
	}


};
/*----------------------------------------------------------------------*/


/*reference: 
https://stackoverflow.com/questions/47981/how-do-you-set-clear-and-toggle-a-single-bit
https://www.learncpp.com/cpp-tutorial/bitwise-operators/
*/
class Vec8b {
public:
	uint8_t data=0;
	MSGPACK_DEFINE_ARRAY(data);

	CUDA_CALLABLE_MEMBER Vec8b() { }
	CUDA_CALLABLE_MEMBER Vec8b( bool v0, bool v1, bool v2=0, bool v3=0, 
								bool v4=0, bool v5=0, bool v6=0, bool v7=0) {
		data |= v0 << 0;
		data |= v1 << 1;
		data |= v2 << 2;
		data |= v3 << 3;
		data |= v4 << 4;
		data |= v5 << 5;
		data |= v6 << 6;
		data |= v7 << 7;
	}
	CUDA_CALLABLE_MEMBER Vec8b(const uint8_t& v) {data = v;}
	CUDA_CALLABLE_MEMBER Vec8b(const unsigned int& v) { data = v; }
	CUDA_CALLABLE_MEMBER Vec8b(const int& v) { data = v; }

	// asignment oprator
	CUDA_CALLABLE_MEMBER Vec8b& operator=(const uint8_t& v) {
		data = v;
		return *this;
	}

	/*assign k-th bit to value */
	inline CUDA_CALLABLE_MEMBER void assignBit(int k, bool value) {
		// clear kth bit      set kth bit
		data = (data & ~(true << k)) | (value << k);
	}

	/*Use bitwise OR operator (|) to set k-th bit to 1*/
	inline CUDA_CALLABLE_MEMBER void setBit(int k) {
		data |= true << k;
	}
	/*Use bitwise AND operator (&) to clear a bit.*/
	inline CUDA_CALLABLE_MEMBER void clearBit(int k) {
		data &= ~(true << k);
	}
	/* check the value of k-th bit*/
	inline CUDA_CALLABLE_MEMBER bool getBit(int k) const {
		return (data >> k) & 1U;
	}
	//CUDA_CALLABLE_MEMBER const double& operator [] (int n) const {

	inline CUDA_CALLABLE_MEMBER friend bool operator==(const Vec8b& v1, const Vec8b& v2) {return v1.data == v2.data;}
	inline CUDA_CALLABLE_MEMBER friend bool operator==(const Vec8b& v1, const uint8_t& v2) { return v1.data == v2; }
	inline CUDA_CALLABLE_MEMBER friend bool operator==(const uint8_t& v1, const Vec8b& v2) { return v2.data == v1; }

	// overloading if (var)
	explicit inline CUDA_CALLABLE_MEMBER operator bool() const {return data;}

	// overwrite bitwise and operator
	inline CUDA_CALLABLE_MEMBER Vec8b operator&(const uint8_t& v2) {return Vec8b(data & v2);}
	inline CUDA_CALLABLE_MEMBER Vec8b operator&(const Vec8b& v2) { return Vec8b(data & v2.data); }
	inline CUDA_CALLABLE_MEMBER friend Vec8b operator&(const uint8_t& v1, const Vec8b& v2) { return Vec8b(v2.data & v1); }

	inline CUDA_CALLABLE_MEMBER Vec8b operator~() { return Vec8b(~data); }


	// convert to bitset
	inline std::bitset<8> toBitset() const { return std::bitset<8>(data); }

	// overloading << operator, least significant bit on the right
	friend std::ostream& operator<<(std::ostream& os, const Vec8b& v){
		os << v.toBitset(); return os;
	}

	inline CUDA_CALLABLE_MEMBER friend Vec3d operator*(const Vec8b& v8b, const Vec3d& v) {
		return Vec3d(v8b.getBit(0) * v.x, v8b.getBit(1) * v.y, v8b.getBit(2) * v.z);
	}
	inline CUDA_CALLABLE_MEMBER friend Vec3d operator*(const Vec3d& v, const Vec8b& v8b) {
		return Vec3d(v8b.getBit(0) * v.x, v8b.getBit(1) * v.y, v8b.getBit(2) * v.z);
	}
};

CUDA_CALLABLE_MEMBER Vec3d& Vec3d::operator*=(const Vec8b& v) {
	x *= v.getBit(0);
	y *= v.getBit(1);
	z *= v.getBit(2);
	return *this;
}

/*----------------------------------------------------------------------*/



/* clamp a value data between lower and upper */
template <typename T>
inline CUDA_CALLABLE_MEMBER void clampInplace(T& n, const T& lower, const T& upper) {
	//assert(lower < upper);
	if (n > upper) { n = upper; }
	else if (n < lower) { n = lower; }
}

/* clamp a value data between lower and upper,
assume periodic between lower and upper */

template <typename T>
void CUDA_CALLABLE_MEMBER clampPeroidicInplace(T& n, const T& lower, const T& upper) {
	//assert(lower < upper);
	if (n > upper) { n = fmod(n - upper, upper - lower) + lower; }
	else if (n < lower) { n = fmod(n - lower, upper - lower) + upper; }
}


#endif
