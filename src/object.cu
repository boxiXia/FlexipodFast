/*modified from the orginal Titan simulation libaray:https://github.com/jacobaustin123/Titan
ref: J. Austin, R. Corrales-Fatou, S. Wyetzner, and H. Lipson, “Titan: A Parallel Asynchronous Library for Multi-Agent and Soft-Body Robotics using NVIDIA CUDA,” ICRA 2020, May 2020.

object.cu defines constraint objects like planes and balls that allow the users
to enforce limitations on movements of objects within the scene.
Generally, an object defines the applyForce method that determines whether to apply a force
to a mass, for example a normal force pushing the mass out of a constaint object or
a frictional force.
*/


#define GLM_FORCE_PURE
#include "object.h"
#include <cmath>


#ifdef GRAPHICS
const glm::vec3 RED(1.0, 0.2, 0.2);
const glm::vec3 GREEN(0.2, 1.0, 0.2);
const glm::vec3 BLUE(0.2, 0.2, 1.0);
const glm::vec3 PURPLE(0.5, 0.2, 0.5);
const glm::vec3 DARKSEAGREEN(0.45, 0.84, 0.5);
const glm::vec3 OLIVEDRAB(0.42, 0.56, 0.14);

#include<glm/gtx/quaternion.hpp> // for rotation
#endif

__device__ const double NORMAL = 1000; // normal force coefficient for contact constraints



//CUDA_CALLABLE_MEMBER CudaBall::CudaBall(const Vec3d & center, double radius) {
//    _center = center;
//    _radius = radius;
//}
//
//CUDA_CALLABLE_MEMBER CudaBall::CudaBall(const Ball & b) {
//    _center = b._center;
//    _radius = b._radius;
//}

CUDA_CALLABLE_MEMBER void CudaBall::applyForce(Vec3d& force, const Vec3d& pos) {
    double dist = (pos - _center).norm();
    if (dist < _radius) {
        force += NORMAL * (pos - _center) / dist;
    }
}

//CUDA_CALLABLE_MEMBER CudaContactPlane::CudaContactPlane(const Vec3d & normal, double offset) {
//    _normal = normal / normal.norm();
//    _offset = offset;
//    _FRICTION_S = 0.0;
//    _FRICTION_K = 0.0;
//}
//
//CudaContactPlane::CudaContactPlane(const ContactPlane & p) {
//    _normal = p._normal;
//    _offset = p._offset;
//
//    _FRICTION_S = p._FRICTION_S;
//    _FRICTION_K = p._FRICTION_K;
//}

//CUDA_CALLABLE_MEMBER void CudaContactPlane::applyForce(Vec3d& force, const Vec3d& pos, const Vec3d& vel) {
//    //    m -> force += (disp < 0) ? - disp * NORMAL * _normal : 0 * _normal; // TODO fix this for the host
//    
//    double disp = _normal.dot(pos) - _offset; // displacement into the plane
//#ifdef __CUDA_ARCH__
//    if(signbit(disp)){ // Determine whether the floating-point value a is negative:https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g2bd7d6942a8b25ae518636dab9ad78a7
//#else
//    if (disp < 0) {// if inside the plane
//#endif
//        Vec3d f_normal = _normal.dot(force) * _normal; // normal force
//
//        if (_FRICTION_S > 0 || _FRICTION_K > 0) {
//            Vec3d v_t = vel - _normal.dot(vel) * _normal; // velocity tangential to the plane
//            double v_t_norm = v_t.norm();
//
//            if (v_t_norm > 1e-10) { // kinetic friction domain
//                double friction_mag = _FRICTION_K * f_normal.norm();
//                force -=  friction_mag / v_t_norm * v_t;
//            }
//            else { // static friction
//                Vec3d f_t = force - f_normal; //  force tangential to the plain
//                if (_FRICTION_S * f_normal.norm() > f_t.norm()) {
//                    force -= f_t;
//                }
//            }
//        }
//        force -= disp * _normal * NORMAL;// displacement force
//    }
//}


CUDA_CALLABLE_MEMBER void CudaContactPlane::applyForce(Vec3d& force, const Vec3d& pos, const Vec3d& vel) {
    //    m -> force += (disp < 0) ? - disp * NORMAL * _normal : 0 * _normal; // TODO fix this for the host

    double disp = _normal.dot(pos) - _offset; // displacement into the plane
#ifdef __CUDA_ARCH__
    if (signbit(disp)) { // Determine whether the floating-point value a is negative:https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g2bd7d6942a8b25ae518636dab9ad78a7
#else
    if (disp < 0) {// if inside the plane
#endif
        ////Vec3d f_normal = _normal.dot(force) * _normal; // normal force (only if infinite stiff)
        Vec3d f_normal = -disp * _normal * NORMAL; // ground spring model
        //if (_FRICTION_S > 0 || _FRICTION_K > 0) {
            Vec3d v_t = vel - _normal.dot(vel) * _normal; // velocity tangential to the plane
            double v_t_norm = v_t.norm();
            if (v_t_norm > 1e-8) { // kinetic friction domain
                //      <----friction magnitude------>   <-friction direction->
                force -= _FRICTION_K * f_normal.norm() / v_t_norm * v_t;
            }
            else { // static friction
                Vec3d f_t = force - f_normal; //  force tangential to the plain
                if (_FRICTION_S * f_normal.norm() > f_t.norm()) {
                    force -= f_t;
                }
                else {// kinetic friction again
                    //       <----friction magnitude------> <- friction direction->
                    force -= _FRICTION_K * f_normal.norm() / v_t_norm * v_t;
                }
            }
        //}
        force -= disp * _normal * NORMAL;// displacement force
    }
    }



#ifdef GRAPHICS

void Ball::normalize(GLfloat * v) {
    GLfloat norm = sqrt(pow(v[0], 2) + pow(v[1], 2) + pow(v[2],2)) / _radius;

    for (int i = 0; i < 3; i++) {
        v[i] /= norm;
    }
}

void Ball::writeTriangle(GLfloat * arr, GLfloat *v1, GLfloat *v2, GLfloat *v3) {
    for (int j = 0; j < 3; j++) {
        arr[j] = v1[j] + _center[j];
    }

    arr += 3;

    for (int j = 0; j < 3; j++) {
        arr[j] = v2[j] + _center[j];
    }

    arr += 3;

    for (int j = 0; j < 3; j++) {
        arr[j] = v3[j] + _center[j];
    }
}

void Ball::subdivide(GLfloat * arr, GLfloat *v1, GLfloat *v2, GLfloat *v3, int depth) {
    GLfloat v12[3], v23[3], v31[3];

    if (depth == 0) {
        writeTriangle(arr, v1, v2, v3);
        return;
    }

    for (int i = 0; i < 3; i++) {
        v12[i] = v1[i]+v2[i];
        v23[i] = v2[i]+v3[i];
        v31[i] = v3[i]+v1[i];
    }

    normalize(v12);
    normalize(v23);
    normalize(v31);

    subdivide(arr, v1, v12, v31, depth - 1);
    arr += 3 * 3 * (int) pow(4, depth - 1);
    subdivide(arr, v2, v23, v12, depth - 1);
    arr += 3 * 3 * (int) pow(4, depth - 1);
    subdivide(arr, v3, v31, v23, depth - 1);
    arr += 3 * 3 * (int) pow(4, depth - 1);
    subdivide(arr, v12, v23, v31, depth - 1);
}


void Ball::generateBuffers() {
    glm::vec3 color = {0.22f, 0.71f, 0.0f};

    GLfloat * vertex_data = new GLfloat[20 * 3 * 3 * (int) pow(4, depth)]; // times 4 for subdivision

    GLfloat X = (GLfloat) _radius * .525731112119133606;
    GLfloat Z = (GLfloat) _radius * .850650808352039932;

    static GLfloat vdata[12][3] = {
            {-X, 0.0, Z}, {X, 0.0, Z}, {-X, 0.0, -Z}, {X, 0.0, -Z},
            {0.0, Z, X}, {0.0, Z, -X}, {0.0, -Z, X}, {0.0, -Z, -X},
            {Z, X, 0.0}, {-Z, X, 0.0}, {Z, -X, 0.0}, {-Z, -X, 0.0}
    };
    static GLuint tindices[20][3] = {
            {0,4,1}, {0,9,4}, {9,5,4}, {4,5,8}, {4,8,1},
            {8,10,1}, {8,3,10}, {5,3,8}, {5,2,3}, {2,7,3},
            {7,10,3}, {7,6,10}, {7,11,6}, {11,0,6}, {0,1,6},
            {6,1,10}, {9,0,11}, {9,11,2}, {9,2,5}, {7,2,11} };

    for (int i = 0; i < 20; i++) {
        subdivide(&vertex_data[3 * 3 * (int) pow(4, depth) * i], vdata[tindices[i][0]], vdata[tindices[i][1]], vdata[tindices[i][2]], depth);
    }

    glGenBuffers(1, &vertices); // create buffer for these vertices
    glBindBuffer(GL_ARRAY_BUFFER, vertices);
    glBufferData(GL_ARRAY_BUFFER, 20 * 3 * 3 * (int) pow(4, depth) * sizeof(GLfloat), vertex_data, GL_STATIC_DRAW);

    GLfloat * color_data = new GLfloat[20 * 3 * 3 * (int) pow(4, depth)]; // TODO constant length array

    for (int i = 0; i < 20 * 3 * (int) pow(4, depth); i++) {
        color_data[3*i] = color[0];
        color_data[3*i + 1] = color[1];
        color_data[3*i + 2] = color[2];
    }

    glGenBuffers(1, &colors);
    glBindBuffer(GL_ARRAY_BUFFER, colors);
    glBufferData(GL_ARRAY_BUFFER, 20 * 3 * 3 * (int) pow(4, depth) * sizeof(GLfloat), color_data, GL_STATIC_DRAW);

    delete [] color_data;
    delete [] vertex_data;

    _initialized = true;
}

void Ball::draw() {
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertices);

    glVertexAttribPointer(
            0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
            3,                  // size
            GL_FLOAT,           // type
            GL_FALSE,           // normalized?
            0,                  // stride
            (void*)0            // array buffer offset
    );

    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, colors);
    glVertexAttribPointer(
            1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
            3,                                // size
            GL_FLOAT,                         // type
            GL_FALSE,                         // normalized?
            0,                                // stride
            (void*)0                          // array buffer offset
    );

    // Draw the triangle !
    glDrawArrays(GL_TRIANGLES, 0, 20 * 3 * (int) pow(4, depth)); // 12*3 indices starting at 0 -> 12 triangles

    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);
}

#endif

#ifdef GRAPHICS

void ContactPlane::generateBuffers() {

    const int radius = 15; // radius [unit] of the plane
    // total 15*15*4*6=5400 points 
    
    // refer to: http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-17-quaternions/
    std::vector<GLfloat> vertex_data;
    std::vector<GLfloat> color_data;

    GLfloat s = 1.f;// scale
    for (int i = -radius; i < radius; i++)
    {
        for (int j = -radius; j < radius; j++)
        {
            GLfloat x = i*s;
            GLfloat y = j*s;
            vertex_data.insert(vertex_data.end(), {
                x,y,0,
                x+s,y+s,0,
                x+s,y,0,
                x,y,0,
                x,y+s,0,
                x+s,y+s,0});//2 triangles of a quad
            // pick one color
            glm::vec3 c = (i + j) % 2 == 0? glm::vec3(0.729f, 0.78f, 0.655f): glm::vec3(0.533f, 0.62f, 0.506f);
            color_data.insert(color_data.end(), {
                c[0],c[1],c[2],
                c[0],c[1],c[2],
                c[0],c[1],c[2],
                c[0],c[1],c[2],
                c[0],c[1],c[2],
                c[0],c[1],c[2]});
        }

    }

    glm::vec3 glm_normal = glm::vec3(_normal[0], _normal[1], _normal[2]);
    auto quat_rot = glm::rotation(glm::vec3(0, 0, 1), glm_normal);

    glm::vec3 glm_offset = (float)_offset*glm_normal;

    #pragma omp parallel for
    for (size_t i = 0; i < vertex_data.size()/3; i++)
    {
        glm::vec3 v(vertex_data[3 * i], vertex_data[3 * i+1], vertex_data[3 * i+2]);
        v = glm::rotate(quat_rot, v) + glm_offset;
        vertex_data[3 * i] = v[0];
        vertex_data[3 * i+1] = v[1];
        vertex_data[3 * i+2] = v[2];
    }

    glGenBuffers(1, &vertices); // create buffer for these vertices
    glBindBuffer(GL_ARRAY_BUFFER, vertices);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)* vertex_data.size(), vertex_data.data(), GL_STATIC_DRAW);


    glGenBuffers(1, &colors);
    glBindBuffer(GL_ARRAY_BUFFER, colors);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * color_data.size(), color_data.data(), GL_STATIC_DRAW);

    _initialized = true;
}

void ContactPlane::draw() {
    // 1st attribute buffer : vertices
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertices);

    glVertexAttribPointer(
            0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
            3,                  // size
            GL_FLOAT,           // type
            GL_FALSE,           // normalized?
            0,                  // stride
            (void*)0            // array buffer offset
    );

    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, colors);
    glVertexAttribPointer(
            1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
            3,                                // size
            GL_FLOAT,                         // type
            GL_FALSE,                         // normalized?
            0,                                // stride
            (void*)0                          // array buffer offset
    );

    // Draw the triangle !
    glDrawArrays(GL_TRIANGLES, 0, 5400); // number of vertices
    
    // Todo: this won't work when the plane is shifted
    //glDrawElements(GL_LINES, 12*6, GL_UNSIGNED_INT, (void*)0); // 3 indices starting at 0 -> 1 triangle

    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);
}
#endif
