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
	Vec v_rot = v * c + cross(k, v) * sin(theta) + dot(k,v) * (1 - c) * k;
	//Vec v_rot = cross(k, v) * sin(theta);
	//v_rot += v * c;
	//v_rot += dot(k, v) * (1 - c) * k;
	v_rot += offset;
	return v_rot;
}