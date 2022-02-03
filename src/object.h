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

    }

    double _radius;
    Vec3d _center;

#ifdef GRAPHICS
    ~Ball() {
        glDeleteBuffers(1, &vertex_buffer);
        glDeleteBuffers(1, &triangle_buffer);
    }

    bool _initialized = false;
    int gl_draw_size;


    GLuint vertex_buffer;
    GLuint triangle_buffer;//index buffer

    void generateBuffers();
    void draw();


    


#endif
};

struct CudaBall {
    //CudaBall() = default;
    //CUDA_CALLABLE_MEMBER CudaBall(const Vec3d & center, double radius);
    //CUDA_CALLABLE_MEMBER CudaBall(const Ball & b);

    __device__ void applyForce(Vec3d& force, const Vec3d& pos, const Vec3d& vel);

    double _radius;
    Vec3d _center;
};

// creates half-space ax + by + cz < d
struct ContactPlane : public Constraint {
    Vec3d _normal;
    double _offset;

    double _FRICTION_K; // kinectic friction coefficient
    double _FRICTION_S; // static friction coefficient



    ContactPlane(const Vec3d & normal, double offset,
        float square_size = 0.5f, float plane_radius = 5) {
        _normal = normal / normal.norm();
        _offset = offset;

        _FRICTION_K = 0.0;
        _FRICTION_S = 0.0;

#ifdef GRAPHICS
        _initialized = false;
        s = square_size;// per square scale (square size [m])
        this->plane_radius = plane_radius; //radius the contact plane 
        nr = int(plane_radius / s);// normalized radius of the plane

#endif
    }


#ifdef GRAPHICS
    ~ContactPlane() {
        glDeleteBuffers(1, &vertex_buffer);
        glDeleteBuffers(1, &triangle_buffer);
    }

    void generateBuffers();
    void draw();

    GLuint vertex_buffer;
    GLuint triangle_buffer;//index buffer

    float s;// per square scale (square size [m])
    float plane_radius; //radius the contact plane
    bool draw_back_face = false; //wether to draw the back face
    int nr;// normalized radius of the plane
    int gl_draw_size; // total number of points

#endif
};

struct CudaContactPlane {

    __device__  void applyForce(Vec3d& force, const Vec3d& pos, const Vec3d& vel);

    __device__  void solveDist(
        Vec3d& pos, 
        Vec3d& pos_prev, 
        Vec3d& vel,
        const Vec3d& global_acc,
        const double dt
    );

    Vec3d _normal;
    double _offset;
    double _FRICTION_K = 0.0;
    double _FRICTION_S = 0.0;
};



// creates terrain with randomized normals over the surface
struct ContactTerrain : public Constraint {

    double _FRICTION_K; // kinectic friction coefficient
    double _FRICTION_S; // static friction coefficient

    ContactTerrain(float unit_size, float terrain_radius, double terrain_waviness) {

        _FRICTION_K = 0.0;
        _FRICTION_S = 0.0;

#ifdef GRAPHICS

        _initialized = false;
        unit = unit_size; // per square scale (square size [m])
        r = terrain_radius; //radius the contact terrain 
        nr = int(r / unit); // normalized radius of the terrain
        waviness = terrain_waviness;

        srand(time(NULL));
        double range = waviness * 2;
        int nd = 2 * nr;

        vertices.resize((nd + 1) * (nd + 1));
        // generate the positions
        for (int i = 0; i <= nd; ++i)
        {
            double x = (i - nr) * unit;
            for (int j = 0; j <= nd; ++j)
            {
                double y = (j - nr) * unit;
                double z = -1.0 + (double)rand() / (double)RAND_MAX * range - waviness;

                vertices[i * (nd + 1) + j] = Vec3d(x, y, z);
            }
        }

        normals.resize(nd * nd * 2);
        // generate the normals
        for (int i = 0; i < nd; ++i)
        {
            for (int j = 0; j < nd; ++j)
            {
                int index = i * (nd + 1) + j;
                Vec3d corners_pos[4] = { vertices[index], vertices[index + nd + 1], vertices[index + nd + 2], vertices[index + 1] };

                Vec3d normal1 = (corners_pos[1] - corners_pos[0]).cross(corners_pos[2] - corners_pos[0]);
                Vec3d normal2 = (corners_pos[2] - corners_pos[0]).cross(corners_pos[3] - corners_pos[0]);

                normals[2 * (i * nd + j)] = normal1;
                normals[2 * (i * nd + j) + 1] = normal2;
            }
        }
#endif
    }

#ifdef GRAPHICS

    ~ContactTerrain() {
        glDeleteBuffers(1, &vertex_buffer);
        glDeleteBuffers(1, &triangle_buffer);
    }

    void generateBuffers();
    void draw();

    GLuint vertex_buffer;
    GLuint triangle_buffer; //index buffer

    float waviness; // the waviness that the terrain may differ from last point
    float unit; // per square scale (square size [m])
    float r; //radius the contact terrain
    bool draw_back_face = false; //wether to draw the back face
    int nr; // normalized radius of the terrain
    int gl_draw_size; // total number of points

    std::vector<Vec3d> vertices; // vertices positions
    std::vector<Vec3d> normals; // normal of each triangle

#endif

};

struct CudaContactTerrain {

    __device__  void applyForce(Vec3d& force, const Vec3d& pos, const Vec3d& vel);

    __device__  void solveDist(
        Vec3d& pos,
        Vec3d& pos_prev,
        Vec3d& vel,
        const Vec3d& global_acc,
        const double dt,
        // Mingxuan Li
        const Vec3d* vertices,
        const Vec3d* normals
    );

    double _FRICTION_K = 0.0;
    double _FRICTION_S = 0.0;

    int _nr;
    double _unit;
    double _waviness;
    Vec3d* _vertices;
    Vec3d* _normals;
};

struct CUDA_GLOBAL_CONSTRAINTS {
    CudaContactPlane* d_planes;
    CudaBall* d_balls;
    CudaContactTerrain* d_terrains;

    size_t num_planes;
    size_t num_balls;
    size_t num_terrains;
};


enum CONSTRAINT_TYPE {
    CONTACT_PLANE, BALL, CONTACT_TERRAIN
};


#endif //TITAN_OBJECT_H
