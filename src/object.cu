//
// Created by Jacob Austin on 5/21/18.
// object.cu defines constraint objects like planes and balls that allow the users
// to enforce limitations on movements of objects within the scene.
// Generally, an object defines the applyForce method that determines whether to apply a force
// to a mass, for example a normal force pushing the mass out of a constaint object or
// a frictional force.

#define GLM_FORCE_PURE
#include "object.h"
#include <cmath>


#ifdef GRAPHICS
const Vec RED(1.0, 0.2, 0.2);
const Vec GREEN(0.2, 1.0, 0.2);
const Vec BLUE(0.2, 0.2, 1.0);
const Vec PURPLE(0.5, 0.2, 0.5);
#endif

__device__ const double NORMAL = 20000; // normal force coefficient for contact constaints



CUDA_CALLABLE_MEMBER CudaBall::CudaBall(const Vec & center, double radius) {
    _center = center;
    _radius = radius;
}

CUDA_CALLABLE_MEMBER CudaBall::CudaBall(const Ball & b) {
    _center = b._center;
    _radius = b._radius;
}

CUDA_CALLABLE_MEMBER void CudaBall::applyForce(Vec& force, Vec& pos) {
    double dist = (pos - _center).norm();
    force += (dist < _radius) ? NORMAL * (pos - _center) / dist : Vec(0, 0, 0);
}

CUDA_CALLABLE_MEMBER CudaContactPlane::CudaContactPlane(const Vec & normal, double offset) {
    _normal = normal / normal.norm();
    _offset = offset;
    _FRICTION_S = 0.0;
    _FRICTION_K = 0.0;
}

CudaContactPlane::CudaContactPlane(const ContactPlane & p) {
    _normal = p._normal;
    _offset = p._offset;

    _FRICTION_S = p._FRICTION_S;
    _FRICTION_K = p._FRICTION_K;
}

CUDA_CALLABLE_MEMBER void CudaContactPlane::applyForce(Vec& force, Vec& pos, Vec& vel) {
    //    m -> force += (disp < 0) ? - disp * NORMAL * _normal : 0 * _normal; // TODO fix this for the host

    double disp = dot(pos, _normal) - _offset; // displacement into the plane
    Vec f_normal = dot(force, _normal) * _normal; // normal force

    if (disp < 0 && (_FRICTION_S > 0 || _FRICTION_K > 0)) { // if inside the plane
        Vec v_perp = vel - dot(vel, _normal) * _normal; // perpendicular velocity
        double v_norm = v_perp.norm();

        if (v_norm > 1e-15) { // kinetic friction domain
            double friction_mag = _FRICTION_K * f_normal.norm();
            force -= v_perp * friction_mag / v_norm;
        } else { // static friction
            Vec f_perp = force - f_normal; // perpendicular force
	        if (_FRICTION_S * f_normal.norm() > f_perp.norm()) {
                force -= f_perp;
	        } 
        }
    }
    // now apply the offset force to push the object out of the plane.
    Vec contact = (disp < 0) ? - disp * NORMAL * _normal : 0 * _normal; // displacement force
    //double f_norm = contact.norm();
    force += contact;

}


//void Container::translate(const Vec & displ) {
//    for (Mass * m : masses) {
//        m -> pos += displ;
//    }
//}

//void Container::rotate(const Vec & axis, double angle) {
//    Vec com(0, 0, 0);
//
//    double total_mass = 0;
//
//    for (Mass * m : masses) {
//        com += m -> m * m -> pos;
//        total_mass += m -> m;
//    }
//
//    com = com / total_mass; // center of mass as centroid
//    Vec temp_axis = axis / axis.norm();
//
//    for (Mass * m : masses) {
//        Vec temp = m -> pos - com; // subtract off center of mass
//        Vec y = temp - dot(temp, temp_axis) * temp_axis; // project onto the given axis and find offset (y coordinate)
//
//        if (y.norm() < 0.0001) { // if on the axis, don't do anything
//            continue;
//        }
//
//        Vec planar(-sin(angle) * y.norm(), cos(angle) * y.norm(), 0); // coordinate in xy space
//        Vec spatial = planar[0] * cross(temp_axis, y / y.norm()) + y / y.norm() * planar[1] + dot(temp, temp_axis) * temp_axis + com; // return to 3D space, then to COM space, then to absolute space
//
//        m -> pos = spatial; // update position
//    }
//}




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
    glm::vec3 color = {0.22f, 0.71f, 0.0f};
    Vec temp = (dot(_normal, Vec(0, 1, 0)) < 0.8) ? Vec(0, 1, 0) : Vec(1, 0, 0);

    Vec v1 = cross(_normal, temp); // two unit vectors along plane
    v1 = v1 / v1.norm();

    Vec v2 = cross(_normal, v1);
    v2 = v2 / v2.norm();

    const static GLfloat vertex_buffer_platform[118] = {
            -1, -1, -1,
            -1, -1,  1,
            -1,  1,  1,
            1,  1, -1,
            -1, -1, -1,
            -1,  1, -1,
            1, -1,  1,
            -1, -1, -1,
            1, -1, -1,
            1,  1, -1,
            1, -1, -1,
            -1, -1, -1,
            -1, -1, -1,
            -1,  1,  1,
            -1,  1, -1,
            1, -1,  1,
            -1, -1,  1,
            -1, -1, -1,
            -1,  1,  1,
            -1, -1,  1,
            1, -1,  1,
            1,  1,  1,
            1, -1, -1,
            1,  1, -1,
            1, -1, -1,
            1,  1,  1,
            1, -1,  1,
            1,  1,  1,
            1,  1, -1,
            -1,  1, -1,
            1,  1,  1,
            -1,  1, -1,
            -1,  1,  1,
            1,  1,  1,
            -1,  1,  1,
            1, -1,  1
    };



    GLfloat vertex_data[108];
    
    for (int i = 0; i < 36; i++) {
        Vec temp = Vec(vertex_buffer_platform[3 * i], vertex_buffer_platform[3 * i + 1], vertex_buffer_platform[3 * i + 2]);
        Vec vertex = 1* (dot(v1, temp) * v1 + dot(v2, temp) * v2 + _normal * (_offset + dot(_normal, temp) - 1.0));

        vertex_data[3 * i] = vertex[0];
        vertex_data[3 * i + 1] = vertex[1];
        vertex_data[3 * i + 2] = vertex[2];
    }

    glGenBuffers(1, &vertices); // create buffer for these vertices
    glBindBuffer(GL_ARRAY_BUFFER, vertices);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_data), vertex_data, GL_STATIC_DRAW);

    GLfloat g_color_buffer_data[108];


    
    for (int i = 0; i < 36; i++) {
        g_color_buffer_data[3 * i] = color[0];
        g_color_buffer_data[3 * i + 1] = color[1];
        g_color_buffer_data[3 * i + 2] = color[2];
    }

    glGenBuffers(1, &colors);
    glBindBuffer(GL_ARRAY_BUFFER, colors);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_color_buffer_data), g_color_buffer_data, GL_STATIC_DRAW);

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
    glDrawArrays(GL_TRIANGLES, 0, 12*3); // 12*3 indices starting at 0 -> 12 triangles
    
    // Todo: this won't work when the plane is shifted
    glDrawElements(GL_LINES, 12*6, GL_UNSIGNED_INT, (void*)0); // 3 indices starting at 0 -> 1 triangle

    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);
}
#endif
