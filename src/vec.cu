//
//  vec.cpp
//  CUDA Physics
//
//  Created by Jacob Austin on 5/13/18.
//  Copyright © 2018 Jacob Austin. All rights reserved.
//

#include "vec.h"


CUDA_DEVICE void Vec::atomicVecAdd(const Vec & v) {
atomicAdd(&data[0], v.data[0]);
atomicAdd(&data[1], v.data[1]);
atomicAdd(&data[2], v.data[2]);
}


CUDA_CALLABLE_MEMBER double dot(const Vec& a, const Vec& b) {

    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

CUDA_CALLABLE_MEMBER Vec cross(const Vec& v1, const Vec& v2) {
    return Vec(v1[1] * v2[2] - v1[2] * v2[1], v2[0] * v1[2] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]);
}