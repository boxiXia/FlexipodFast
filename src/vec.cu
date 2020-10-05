/*modified from the orginal Titan simulation libaray:https://github.com/jacobaustin123/Titan
ref: J. Austin, R. Corrales-Fatou, S. Wyetzner, and H. Lipson, “Titan: A Parallel Asynchronous Library for Multi-Agent and Soft-Body Robotics using NVIDIA CUDA,” ICRA 2020, May 2020.
*/
#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#endif // !CUDA_API_PER_THREAD_DEFAULT_STREAM

#include "vec.h"

// https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
/* rotate a vector {v_} with rotation axis {k} anchored at point {offset} by {theta} [rad]
   k is a (unit) direction vector */
CUDA_CALLABLE_MEMBER Vec3d AxisAngleRotaion(const Vec3d& k, const Vec3d& v_, const double& theta, const Vec3d& offset) {
	Vec3d v = v_ - offset;
	double c = cos(theta);
	//Vec3d v_rot = v * c + cross(k, v) * sin(theta) + dot(k,v) * (1 - c) * k;
	Vec3d v_rot = cross(k, v);
	v_rot *= sin(theta);
	v_rot += v * c;
	v_rot += dot(k, v) * (1 - c) * k;

	v_rot += offset;
	return v_rot;
}

// https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
/* rotate a vector {v_} with rotation axis {axis_end-axis_start} by {theta} [rad] */
CUDA_CALLABLE_MEMBER Vec3d AxisAngleRotaion(const Vec3d& axis_start,const Vec3d& axis_end, const Vec3d& v_, const double& theta) {
	Vec3d k = (axis_end - axis_start).normalize(); // rotation axis{ k } is a (unit) direction vector
	Vec3d v = v_ - axis_start;
	double c = cos(theta);
	//Vec3d v_rot = v * c + cross(k, v) * sin(theta) + dot(k,v) * (1 - c) * k;
	Vec3d v_rot = cross(k, v);
	v_rot *= sin(theta);
	v_rot += v * c;
	v_rot += dot(k, v) * (1 - c) * k;

	v_rot += axis_start;
	return v_rot;
}

CUDA_CALLABLE_MEMBER Vec3d lerp(Vec3d p0, Vec3d p1, double t) {
	// Vec3d p_lerp = (p1-p0)*t + p0;
	Vec3d p_lerp = p1; 
	p_lerp -=p0;
	p_lerp *= t;
	p_lerp += p0;
	return p_lerp;
}


CUDA_CALLABLE_MEMBER Vec3d slerp(Vec3d p0, Vec3d p1, double t) {
	double w = angleBetween(p0, p1);//total angle
	double s = sin(w);
	//fixed numerical instability
	Vec3d p_lerp = abs(s)>1e-10? sin((1 - t) * w) / s * p0 + sin(t * w) / s * p1 : p1;
	return p_lerp;
}

// https://stackoverflow.com/questions/14066933/direct-way-of-computing-clockwise-angle-between-2-vectors
/*compute the clockwise angle between p0 and p1 robtated about the normal, NOTE: normal must be normalized!*/
CUDA_CALLABLE_MEMBER double signedAngleBetween(Vec3d p0, Vec3d p1, Vec3d normal) {
	return atan2(cross(p0, p1).dot(normal), p0.dot(p1));
}
