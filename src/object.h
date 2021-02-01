/*modified from the orginal Titan simulation libaray:https://github.com/jacobaustin123/Titan
ref: J. Austin, R. Corrales-Fatou, S. Wyetzner, and H. Lipson, “Titan: A Parallel Asynchronous Library for Multi-Agent and Soft-Body Robotics using NVIDIA CUDA,” ICRA 2020, May 2020.
*/

#ifndef TITAN_OBJECT_H
#define TITAN_OBJECT_H

//#include "mass.h"
//#include "spring.h"
#include "vec.h"

#ifdef GRAPHICS
// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h> 

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#endif

#include <vector>

#include <thrust/device_vector.h>


struct CUDA_MASS;
class Spring;
class Mass;

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

class Constraint { // constraint like plane or sphere which applies force to masses
public:
    virtual ~Constraint() = default;

#ifdef GRAPHICS
    bool _initialized;
    virtual void generateBuffers() = 0;
    virtual void draw() = 0;
#endif
};

struct Ball : public Constraint {
    Ball(const Vec3d & center, double radius) {
        _center = center;
        _radius = radius;

#ifdef GRAPHICS
        _initialized = false;
#endif

    }

    double _radius;
    Vec3d _center;

#ifdef GRAPHICS
    ~Ball() {
        glDeleteBuffers(1, &vertex_buffer);
        glDeleteBuffers(1, &color_buffer);
        glDeleteBuffers(1, &normal_buffer);
    }
    GLuint vertex_buffer;
    GLuint color_buffer;
    GLuint normal_buffer;

    void generateBuffers();
    void draw();

    void subdivide(GLfloat * arr, GLfloat *v1, GLfloat *v2, GLfloat *v3, int depth);
    void writeTriangle(GLfloat * arr, GLfloat *v1, GLfloat *v2, GLfloat *v3);
    void normalize(GLfloat * v);

    int depth = 2;


#endif
};

struct CudaBall {
    //CudaBall() = default;
    //CUDA_CALLABLE_MEMBER CudaBall(const Vec3d & center, double radius);
    //CUDA_CALLABLE_MEMBER CudaBall(const Ball & b);

    __device__ void applyForce(Vec3d& force, const Vec3d& pos);

    double _radius;
    Vec3d _center;
};

// creates half-space ax + by + cz < d
struct ContactPlane : public Constraint {
    Vec3d _normal;
    double _offset;

    double _FRICTION_K; // kinectic friction coefficient
    double _FRICTION_S; // static friction coefficient

    ContactPlane(const Vec3d & normal, double offset) {
        _normal = normal / normal.norm();
        _offset = offset;

        _FRICTION_K = 0.0;
        _FRICTION_S = 0.0;

#ifdef GRAPHICS
        _initialized = false;
#endif
    }


#ifdef GRAPHICS
    ~ContactPlane() {
        glDeleteBuffers(1, &vertex_buffer);
        glDeleteBuffers(1, &color_buffer);
    }

    void generateBuffers();
    void draw();

    GLuint vertex_buffer;
    GLuint color_buffer;
    GLuint normal_buffer;


#endif
};

struct CudaContactPlane {

    __device__  void applyForce(Vec3d& force, const Vec3d& pos, const Vec3d& vel);

    Vec3d _normal;
    double _offset;
    double _FRICTION_K = 0.0;
    double _FRICTION_S = 0.0;
};



struct CUDA_GLOBAL_CONSTRAINTS {
    CudaContactPlane * d_planes;
    CudaBall * d_balls;

    size_t num_planes;
    size_t num_balls;
};


enum CONSTRAINT_TYPE {
    CONTACT_PLANE, BALL
};


#endif //TITAN_OBJECT_H
