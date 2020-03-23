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

#ifdef CONSTRAINTS
#include <thrust/device_vector.h>
#endif

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
    Ball(const Vec & center, double radius) {
        _center = center;
        _radius = radius;

#ifdef GRAPHICS
        _initialized = false;
#endif

    }

    double _radius;
    Vec _center;

#ifdef GRAPHICS
    ~Ball() {
        glDeleteBuffers(1, &vertices);
        glDeleteBuffers(1, &colors);
    }

    void generateBuffers();
    void draw();

    void subdivide(GLfloat * arr, GLfloat *v1, GLfloat *v2, GLfloat *v3, int depth);
    void writeTriangle(GLfloat * arr, GLfloat *v1, GLfloat *v2, GLfloat *v3);
    void normalize(GLfloat * v);

    int depth = 2;

    GLuint vertices;
    GLuint colors;
#endif
};

struct CudaBall {
    CudaBall() = default;
    CUDA_CALLABLE_MEMBER CudaBall(const Vec & center, double radius);
    CUDA_CALLABLE_MEMBER CudaBall(const Ball & b);

    CUDA_CALLABLE_MEMBER void applyForce(Vec& force, Vec& pos);

    double _radius;
    Vec _center;
};

// creates half-space ax + by + cz < d
struct ContactPlane : public Constraint {
    ContactPlane(const Vec & normal, double offset) {
        _normal = normal / normal.norm();
        _offset = offset;

        _FRICTION_K = 0.0;
        _FRICTION_S = 0.0;

#ifdef GRAPHICS
        _initialized = false;
#endif
    }

    Vec _normal;
    double _offset;

    double _FRICTION_K;
    double _FRICTION_S;

#ifdef GRAPHICS
    ~ContactPlane() {
        glDeleteBuffers(1, &vertices);
        glDeleteBuffers(1, &colors);
    }

    void generateBuffers();
    void draw();

    GLuint vertices;
    GLuint colors;
#endif
};

struct CudaContactPlane {
    CudaContactPlane() = default;
    CUDA_CALLABLE_MEMBER CudaContactPlane(const Vec & normal, double offset);
    CudaContactPlane(const ContactPlane & p);

    CUDA_CALLABLE_MEMBER void applyForce(Vec& force, Vec& pos, Vec& vel);

    Vec _normal;
    double _offset;
    double _FRICTION_K;
    double _FRICTION_S;
};



struct CUDA_GLOBAL_CONSTRAINTS {
    CudaContactPlane * d_planes;
    CudaBall * d_balls;

    int num_planes;
    int num_balls;
};



#ifdef CONSTRAINTS
enum CONSTRAINT_TYPE {
    CONTACT_PLANE, BALL
};
#endif




#endif //TITAN_OBJECT_H
