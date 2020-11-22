#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#endif // !CUDA_API_PER_THREAD_DEFAULT_STREAM

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

#include<cuda_runtime.h>
#include<cuda_device_runtime_api.h>
#include<device_launch_parameters.h>

#include <chrono> // for time measurement

#include<algorithm>

#define _USE_MATH_DEFINES
#include <math.h>

#include <omp.h>

//#include <msgpack.hpp>

#include <thread>

#include "vec.h"

int main() {
	Mat3d a(1, 2, 3, 4, 5, 6, 7, 8, 9);
	Mat3d b(a);
	Vec3d v1(1, 2, 3);
	Vec3d v2(4, 5, 6);
	Vec3d v3(7, 8, 9);
	Mat3d c(v1, v2, v3);// default construct from row vector
	Mat3d d(v1, v2, v3, false);//constuct from column vector
	assert(c.getRow(0) == v1);
	assert(c.getRow(1) == v2);
	assert(c.getRow(2) == v3);
	assert(d.getColumn(0) == v1);
	assert(d.getColumn(1) == v2);
	assert(d.getColumn(2) == v3);

	assert(c.dot(d) == Mat3d(14, 32, 50, 32, 77, 122, 50, 122, 194));
	assert(c.transpose() == d);
	//std::cout << a;
	//a.print();
	assert(a == b);
	assert(a == c);
	assert((Mat3d(a) += b) == Mat3d(2, 4, 6, 8, 10, 12, 14, 16, 18));
	assert((Mat3d(a) += 5) == Mat3d(6, 7, 8, 9, 10, 11, 12, 13, 14));

	assert((Mat3d(a) -= b) == Mat3d(0, 0, 0, 0, 0, 0, 0, 0, 0));
	assert((Mat3d(a) -= 5) == Mat3d(-4, -3, -2, -1, 0, 1, 2, 3, 4));
	assert((Mat3d(a) *= b) == Mat3d(1, 4, 9, 16, 25, 36, 49, 64, 81));
	assert((Mat3d(a) *= 5) == Mat3d(5, 10, 15, 20, 25, 30, 35, 40, 45));
	assert((Mat3d(a) /= b) == Mat3d(1., 1., 1., 1., 1., 1., 1., 1., 1.));
	assert((Mat3d(a) /= 5) == Mat3d(0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8));
	assert(-a == Mat3d(-1, -2, -3, -4, -5, -6, -7, -8, -9));


	assert(a + b == Mat3d(2, 4, 6, 8, 10, 12, 14, 16, 18));
	assert(a + 5 == Mat3d(6, 7, 8, 9, 10, 11, 12, 13, 14));
	assert(5 + a == Mat3d(6, 7, 8, 9, 10, 11, 12, 13, 14));
	assert(a - b == Mat3d(0, 0, 0, 0, 0, 0, 0, 0, 0));
	assert(a - 5 == Mat3d(-4, -3, -2, -1, 0, 1, 2, 3, 4));
	assert(5 - a == Mat3d(4, 3, 2, 1, 0, -1, -2, -3, -4));
	assert(a * b == Mat3d(1, 4, 9, 16, 25, 36, 49, 64, 81));
	assert(a * 5 == Mat3d(5, 10, 15, 20, 25, 30, 35, 40, 45));
	assert(5 * a == Mat3d(5, 10, 15, 20, 25, 30, 35, 40, 45));
	assert(a / 5 == Mat3d(0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8));
	assert(5 / a == Mat3d(5., 2.5, 5. / 3., 1.25, 1., 5. / 6, 5. / 7, 0.625, 5. / 9));
	assert(a / b == Mat3d(1., 1., 1., 1., 1., 1., 1., 1., 1.));
	assert(dot(a, Mat3d::identity()) == a);
	assert(dot(a, b) == Mat3d(30, 36, 42, 66, 81, 96, 102, 126, 150));
	assert(a.transpose() == Mat3d(1, 4, 7, 2, 5, 8, 3, 6, 9));
	assert(a.transpose().dot(b) == Mat3d(66, 78, 90, 78, 93, 108, 90, 108, 126));
	assert(a.det() == 0);//if not than <1e-15
	Mat3d e(1, 2, 3, 7, 6, 8, 2, 7, 3);
	assert(e.det() == 63.0);
	assert(a.trace() == 15);
	//std::cout << a * b;
	//std::cout << a.dot(b);
	Mat3d f = e.inv() * e.det();
	//std::cout << e.inv()*e.det();
	//assert(abs((e.inv() * e.det() - Mat3d(-38., 15., -2., -5., -3., 13., 37., -3., -8.)).det())<1e-15);
	assertClose(e.inv() * e.det(), Mat3d(-38., 15., -2., -5., -3., 13., 37., -3., -8.));
	
	
	assert(a.dot(v1) == dot(a, v1));
	assert(a.dot(v1) == Vec3d(14, 32, 50));
	//std::cout << 1.0 / 0.0;


	Mat3d tmp = Mat3d::fromRotVec(Vec3d(0, 0, 1) * M_PI_2);
	assert(abs((Mat3d::fromRotVec(Vec3d(0, 0, 1) * M_PI_2)- Mat3d(0., -1., 0., 1., 0., 0., 0., 0., 1.)).det())<1e-15);

	{
		double dt = 0.001;
		Vec3d v0(0, 0, 1);
		v0.normalize()*=M_PI_4;
		Vec3d v1(0, 1, 0);
		//v1.normalize();
		v1 *= dt;
		Mat3d r0 = Mat3d::fromRotVec(v0);
		std::cout << r0;
		Mat3d r1 = Mat3d::fromRotVec(v1).dot(r0);
		std::cout << r1;

		Vec3d w_a = Mat3d::angularVelocityFromRotation(r0, r1, dt,true);
		std::cout << w_a << std::endl;

	}
}