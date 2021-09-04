/*modified from the orginal Titan simulation libaray:https://github.com/jacobaustin123/Titan
ref: J. Austin, R. Corrales-Fatou, S. Wyetzner, and H. Lipson, “Titan: A Parallel Asynchronous Library for Multi-Agent and Soft-Body Robotics using NVIDIA CUDA,” ICRA 2020, May 2020.
*/

#ifdef GRAPHICS
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>
using namespace std;

#include <stdlib.h>
#include <string.h>

#include "shader.h"


// spherical linear interpolation
glm::vec3 slerp(glm::vec3 p0, glm::vec3 p1, float t) {
    // angle between p0 and p1
    float w = acosf(glm::dot(p0, p1) / (glm::l2Norm(p0) * glm::l2Norm(p1)));
    float s = sinf(w);
    //fixed numerical instability
    glm::vec3 p_lerp = abs(s) > 1e-8 ? sinf((1 - t) * w) / s * p0 + sinf(t * w) / s * p1 : p1;
    return p_lerp;
}

void Camera::follow(const glm::vec3& target, const float interp_factor) {
    if (should_follow) {
        // desired camera position
        glm::vec3 href = getHorizontalDirection();
        glm::vec3 pos_desired = target +
            h_offset * glm::rotate(href, yaw, up) + up_offset * up;

        // interpolate position
        pos = glm::mix(pos, pos_desired, interp_factor);

        // interpolate the orientation // https://en.wikipedia.org/wiki/Slerp
        glm::vec3 dir_desired = glm::normalize(target - pos);
        dir = glm::normalize(slerp(dir, dir_desired, interp_factor));
    }
}

glm::vec3 Camera::getHorizontalDirection() {
    glm::vec3 href(1, 0, 0); // camera horizontal reference direction
    if (abs(glm::dot(href, up)) > 0.9) { href = glm::vec3(0, 1, 0); }
    href = glm::normalize(href - glm::dot(href, up) * up);
    return href;
}

#endif