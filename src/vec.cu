//
//  vec.cpp
//  CUDA Physics
//
//  Created by Jacob Austin on 5/13/18.
//  Copyright © 2018 Jacob Austin. All rights reserved.
//

#include "vec.h"

// https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
// rotate a vector {v_} with rotation axis {k} anchored at point {offset} by {theta} [rad]
CUDA_CALLABLE_MEMBER Vec AxisAngleRotaion(const Vec& k, const Vec& v_, const double& theta, const Vec& offset) {
	Vec v = v_ - offset;
	double c = cos(theta);
	//Vec v_rot = v * c + cross(k, v) * sin(theta) + dot(k,v) * (1 - c) * k;
	Vec v_rot = cross(k, v);
	v_rot *= sin(theta);
	v_rot += v * c;
	v_rot += dot(k, v) * (1 - c) * k;
	v_rot += offset;
	return v_rot;
}


CUDA_CALLABLE_MEMBER Vec slerp(Vec p0, Vec p1, double t) {
	double w = angleBetween(p0, p1);//total angle
	double s = sin(w);
	Vec p_lerp = sin((1 - t) * w) / s * p0 + sin(t * w) / s * p1;
	return p_lerp;
}

// https://stackoverflow.com/questions/14066933/direct-way-of-computing-clockwise-angle-between-2-vectors
CUDA_CALLABLE_MEMBER double signedAngleBetween(Vec p0, Vec p1, Vec normal) {
	return atan2(cross(p0, p1).dot(normal), p0.dot(p1));
}

//CUDA_DEVICE void Vec::atomicVecAdd(const Vec& v) {
//#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
//	atomicAdd(&x, v.x);
//	atomicAdd(&y, v.y);
//	atomicAdd(&z, v.z);
//#elif defined(__CUDA_ARCH__) &&__CUDA_ARCH__ < 600
//	atomicDoubleAdd(&x, v.x);
//	atomicDoubleAdd(&y, v.y);
//	atomicDoubleAdd(&z, v.z);
//#endif
//}