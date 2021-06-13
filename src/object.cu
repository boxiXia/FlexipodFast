/*modified from the orginal Titan simulation libaray:https://github.com/jacobaustin123/Titan
ref: J. Austin, R. Corrales-Fatou, S. Wyetzner, and H. Lipson, “Titan: A Parallel Asynchronous Library for Multi-Agent and Soft-Body Robotics using NVIDIA CUDA,” ICRA 2020, May 2020.

object.cu defines constraint objects like planes and balls that allow the users
to enforce limitations on movements of objects within the scene.
Generally, an object defines the applyForce method that determines whether to apply a force
to a mass, for example a normal force pushing the mass out of a constaint object or
a frictional force.
*/
#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#endif // !CUDA_API_PER_THREAD_DEFAULT_STREAM

#define GLM_FORCE_PURE
#include "object.h"
#include <cmath>

// for unordered map with pair as hash
#include <unordered_map>
#include <functional>

#ifdef GRAPHICS
const glm::vec3 RED(1.0, 0.2, 0.2);
const glm::vec3 GREEN(0.2, 1.0, 0.2);
const glm::vec3 BLUE(0.2, 0.2, 1.0);
const glm::vec3 PURPLE(0.5, 0.2, 0.5);
const glm::vec3 DARKSEAGREEN(0.45, 0.84, 0.5);
const glm::vec3 OLIVEDRAB(0.42, 0.56, 0.14);

struct VERTEX_DATA {
    glm::vec3 pos; // 0: vertex position
    glm::vec3 color; // 1: vertex color
    glm::vec3 normal; //3: vertex normal
};

#include<glm/gtx/quaternion.hpp> // for rotation
#endif

//__device__ const double K_NORMAL = 100; // normal force coefficient for contact constraints
//__device__ const double DAMPING_NORMAL = 3; // normal damping coefficient per kg mass
__device__ const double K_NORMAL = 800; // normal force coefficient for contact constraints
__device__ const double DAMPING_NORMAL = 1; // normal damping coefficient per kg mass


__device__ void CudaBall::applyForce(Vec3d& force, const Vec3d& pos, const Vec3d& vel) {
    Vec3d d = (pos - _center);
    double d_norm = d.norm();
    double disp = d_norm - _radius;

    if (disp < 0) {
        Vec3d _normal = d / d_norm; //todo too small case?
        Vec3d fc= -disp * _normal * K_NORMAL;

        double vn_s = _normal.dot(vel); // velocity (scalar) normal to the sphere
        Vec3d vn = vn_s * _normal; // velocity normal to the sphere

        fc -= (vn_s < 0) * vn * DAMPING_NORMAL;//TODO damping may be greater than total force
        force += fc;
    }
}


__device__ void CudaContactPlane::applyForce(Vec3d& force, const Vec3d& pos, const Vec3d& vel) {
    //    m -> force += (disp < 0) ? - disp * K_NORMAL * _normal : 0 * _normal; // TODO fix this for the host

    double disp = _normal.dot(pos) - _offset; // displacement into the plane
#ifdef __CUDA_ARCH__
    if (signbit(disp)) { // Determine whether the floating-point value a is negative:https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html#group__CUDA__MATH__DOUBLE_1g2bd7d6942a8b25ae518636dab9ad78a7
#else
    if (disp < 0) {// if inside the plane
#endif
		////Vec3d f_normal = _normal.dot(force) * _normal; // normal force (only if infinite stiff)
        // fc is the constraint force
        Vec3d fc = -disp * _normal * K_NORMAL; // first part: ground reaction force normal to the ground, ground spring model
        double fn_ground_norm = fc.norm(); // ground reaction force scalar
        double vn_s = _normal.dot(vel); // velocity (scalar) normal to the plane
		Vec3d vn = vn_s * _normal; // velocity normal to the plane
		Vec3d vt = vel - vn; // velocity tangential to the plane
		double vt_norm = vt.norm();
		if (vt_norm > 1e-13) { // kinetic friction domain
			//      <----friction magnitude------>   <-friction direction->
            fc -= _FRICTION_K * fn_ground_norm / vt_norm * vt;
		}
		else { // static friction
			Vec3d fn = force.dot(_normal) * _normal; // force normal to the plane
			Vec3d ft = force - fn; // force tangential to the plain
			float ft_norm = ft.norm();//force tangential to the plain (mangitude)
			if (_FRICTION_S * fn_ground_norm > ft_norm) {
                fc -= ft;
			}
			else {// kinetic friction again
				//       <----friction magnitude------> <- friction direction->
                fc -= _FRICTION_K * fn_ground_norm / ft_norm * ft;
			}
		}
		fc -= (vn_s < 0)*vn * DAMPING_NORMAL;//TODO damping may be greater than total force
        force += fc;
	}
}



#ifdef GRAPHICS

using TriangleList = std::vector<glm::uvec3>; // triangle indices
using VertexList = std::vector<glm::vec3>;

/*start with a hard-coded indexed-mesh representation of the icosahedron
ref: https://schneide.blog/2016/07/15/generating-an-icosphere-in-c/
*/
namespace icosahedron
{
    const float X = .525731112119133606f;
    const float Z = .850650808352039932f;
    const float N = 0.f;

    static const VertexList vertex =
    {
      {-X,N,Z}, {X,N,Z}, {-X,N,-Z}, {X,N,-Z},
      {N,Z,X}, {N,Z,-X}, {N,-Z,X}, {N,-Z,-X},
      {Z,X,N}, {-Z,X, N}, {Z,-X,N}, {-Z,-X, N}
    };

    //static const TriangleList triangle =
    //{
    //  {0,4,1},{0,9,4},{9,5,4},{4,5,8},{4,8,1},
    //  {8,10,1},{8,3,10},{5,3,8},{5,2,3},{2,7,3},
    //  {7,10,3},{7,6,10},{7,11,6},{11,0,6},{0,1,6},
    //  {6,1,10},{9,0,11},{9,11,2},{9,2,5},{7,2,11}
    //}; // if front face is clockwise
    static const TriangleList triangle =
    {
      {0,1,4},{0,4,9},{9,4,5},{4,8,5},{4,1,8},
      {8,1,10},{8,10,3},{5,8,3},{5,3,2},{2,3,7},
      {7,3,10},{7,10,6},{7,6,11},{11,6,0},{0,6,1},
      {6,10,1},{9,11,0},{9,2,11},{9,5,2},{7,11,2}
    }; // if front face is counterclockwise
}


/* from boost::hash_combine
ref: https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x */
template <class T>
inline void hash_combine(std::size_t& seed, const T& v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

/* A hash function used to hash a pair of any kind 
ref: https://stackoverflow.com/questions/32685540/why-cant-i-compile-an-unordered-map-with-a-pair-as-key */
struct pair_hash {
    template <class T1, class T2>
    size_t operator()(const std::pair<T1, T2>& p) const
    {
        std::size_t h1 = std::hash<T1>{}(p.first);
        hash_combine(h1, p.second);
        return h1;
    }
};

//using Lookup = std::map<std::pair<int, int>, int>; // less efficient than unordered_map
using Lookup = std::unordered_map<std::pair<int, int>, int, pair_hash>;
/*Each edge in the old model is subdivided and the resulting vertex is moved on
  to the unit sphere by normalization. The key here is to not duplicate the newly
  created vertices. This is done by keeping a lookup of the edge to the new vertex
  it generates. Note that the orientation of the edge does not matter here, so we
  need to normalize the edge direction for the lookup. We do this by forcing the
  lower index first. Here’s the code that either creates or reused the vertex for
  a single edge*/
int vertexForEdge(Lookup& lookup, VertexList& vertices, int first, int second)
{
    Lookup::key_type key(first, second);
    if (key.first > key.second)
        std::swap(key.first, key.second);

    auto inserted = lookup.insert({ key, vertices.size() });
    if (inserted.second)
    {
        auto& edge0 = vertices[first];
        auto& edge1 = vertices[second];
        auto point = glm::normalize(edge0 + edge1);
        vertices.push_back(point);
    }

    return inserted.first->second;
}

TriangleList subdivide(VertexList& vertices, TriangleList triangles)
{
    Lookup lookup;
    TriangleList result;

    for (auto&& each : triangles)
    {
        std::array<int, 3> mid;
        for (int edge = 0; edge < 3; ++edge)
        {
            mid[edge] = vertexForEdge(lookup, vertices,
                each[edge], each[(edge + 1) % 3]);
        }
        result.push_back({ each[0], mid[0], mid[2] });
        result.push_back({ each[1], mid[1], mid[0] });
        result.push_back({ each[2], mid[2], mid[1] });
        result.push_back({ mid[0], mid[1], mid[2] });
    }
    return result;
}

using IndexedMesh = std::pair<VertexList, TriangleList>;

IndexedMesh makeIcosphere(int subdivisions)
{
    VertexList vertex = icosahedron::vertex;
    TriangleList triangle = icosahedron::triangle;

    for (int i = 0; i < subdivisions; ++i)
    {
        triangle = subdivide(vertex, triangle);
    }
    return{ vertex, triangle };
}

void Ball::generateBuffers() {

    int subdivisions = 3;
    glm::vec3 color = { 0.22f, 0.71f, 0.0f };

    VertexList vertex = icosahedron::vertex;
    TriangleList triangle = icosahedron::triangle;

    for (int i = 0; i < subdivisions; ++i)
    {
        triangle = subdivide(vertex, triangle);
    }

    int num_vertex = vertex.size();
    std::vector<VERTEX_DATA> vertex_data(num_vertex);

    glm::vec3 center = { _center.x, _center.y, _center.z };
    float radius = (float)_radius;
    for (int i = 0; i < num_vertex; i++)
    {
        vertex_data[i].pos = radius*vertex[i] + center;
        vertex_data[i].color = color;
        vertex_data[i].normal = vertex[i];
    }

    gl_draw_size = triangle.size() * 3;

    glGenBuffers(1, &vertex_buffer); // create buffer for these vertices
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(VERTEX_DATA) * vertex_data.size(), vertex_data.data(), GL_DYNAMIC_DRAW);

    glGenBuffers(1, &triangle_buffer); // buffer for the triangle
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangle_buffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(decltype(triangle)) * triangle.size(), triangle.data(), GL_DYNAMIC_DRAW);

    // (optional) unbind to avoid accidental modification of the buffers
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    _initialized = true;
}

void Ball::draw() {
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    // 1st attribute buffer : vertices
    glVertexAttribPointer(
        0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
        3,                  // size
        GL_FLOAT,           // type
        GL_FALSE,           // normalized?
        sizeof(VERTEX_DATA),// stride
        (void*)0            // array buffer offset
    );
    glEnableVertexAttribArray(0);

    //color;
    glVertexAttribPointer(
        1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
        3,                                // size
        GL_FLOAT,                         // type
        GL_FALSE,                         // normalized?
        sizeof(VERTEX_DATA),              // stride
        (void*)(sizeof(VERTEX_DATA::pos)) // array buffer offset
    );
    glEnableVertexAttribArray(1);

    // normal;
    glVertexAttribPointer(
        2,                                // attribute. No particular reason for 1, but must match the layout in the shader.
        3,                                // size
        GL_FLOAT,                         // type
        GL_FALSE,                         // normalized?
        sizeof(VERTEX_DATA),              // stride
        (void*)(sizeof(VERTEX_DATA::pos) + (sizeof(VERTEX_DATA::color)))// array buffer offset
    );
    glEnableVertexAttribArray(2);

    // Draw the triangle !
    //glDrawArrays(GL_TRIANGLES, 0, gl_draw_size); // number of vertices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangle_buffer);
    glDrawElements(GL_TRIANGLES, gl_draw_size, GL_UNSIGNED_INT, (void*)0);

    // (optional) unbind to avoid accidental modification of the buffers
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
}
#endif

#ifdef GRAPHICS

void ContactPlane::generateBuffers() {    
    // refer to: http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-17-quaternions/

    int num_square = 2 * nr * 2 * nr;
    int num_vertex = num_square * 4;
    gl_draw_size = num_square * 6 *(draw_back_face? 2:1);
    
    std::vector<VERTEX_DATA> vertex_data(num_vertex);
    std::vector<GLuint> triangle(gl_draw_size);//triangle indices

    //vertices of a square
    glm::vec3 square[4] = {{0,0,0},{s,0,0},{s,s,0},{0,s,0}};

    GLuint square_triangle[6] = { 0,1,2,0,2,3 };//indices of a square counterclockwise

    glm::vec3 glm_normal = glm::vec3(_normal.x, _normal.y, _normal.z);
    auto quat_rot = glm::rotation(glm::vec3(0, 0, 1), glm_normal);
    glm::vec3 glm_offset = (float)_offset * glm_normal;

	int nd = 2 * nr; // normalized plane diameter

#pragma omp parallel for
	for (int idx = 0; idx < num_square; idx++)
	{
		int i = idx / nd;
		int j = idx % nd;
		GLfloat x = (i - nr) * s;
		GLfloat y = (j - nr) * s;
		int vert_start = 4 * idx; // start index of the vertex
		// pick one color
		glm::vec3 c = (i + j) % 2 == 0 ? glm::vec3(0.729f, 0.78f, 0.655f) : glm::vec3(0.533f, 0.62f, 0.506f);
		for (int k = 0; k < 4; k++) //2 triangles of a quad
		{
			int vid = vert_start + k; //vertex index
			vertex_data[vid].pos = glm::rotate(quat_rot, glm::vec3(x, y, 0) + square[k]) + glm_offset;
			vertex_data[vid].normal = glm_normal;
			vertex_data[vid].color = c;
		}
		// triangles
		int triag_start = 6 * idx;
		for (int k = 0; k < 6; k++) //2 triangles of a quad
		{
			triangle[triag_start + k] = vert_start + square_triangle[k];
		}
    }
    if (draw_back_face) {//draw front and back
        int offset = gl_draw_size / 2;
        for (int i = 0; i < offset; i++)
        {
            triangle[offset + i] = triangle[offset - i-1];
        }
    }

    glGenBuffers(1, &vertex_buffer); // create buffer for these vertices
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(VERTEX_DATA)* vertex_data.size(), vertex_data.data(), GL_DYNAMIC_DRAW);
    
    glGenBuffers(1, &triangle_buffer); // buffer for the triangle
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangle_buffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * triangle.size(), triangle.data(), GL_DYNAMIC_DRAW);
    
    // (optional) unbind to avoid accidental modification of the buffers
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    _initialized = true;
}

void ContactPlane::draw() {
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    // 1st attribute buffer : vertices
    glVertexAttribPointer(
            0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
            3,                  // size
            GL_FLOAT,           // type
            GL_FALSE,           // normalized?
            sizeof(VERTEX_DATA),// stride
            (void*)0            // array buffer offset
    );
    glEnableVertexAttribArray(0);
   
    //color;
    glVertexAttribPointer(
            1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
            3,                                // size
            GL_FLOAT,                         // type
            GL_FALSE,                         // normalized?
            sizeof(VERTEX_DATA),              // stride
            (void*)(sizeof(VERTEX_DATA::pos)) // array buffer offset
    );
    glEnableVertexAttribArray(1);

    // normal;
    glVertexAttribPointer(
        2,                                // attribute. No particular reason for 1, but must match the layout in the shader.
        3,                                // size
        GL_FLOAT,                         // type
        GL_FALSE,                         // normalized?
        sizeof(VERTEX_DATA),              // stride
        (void*)(sizeof(VERTEX_DATA::pos)+ (sizeof(VERTEX_DATA::color)))// array buffer offset
    );
    glEnableVertexAttribArray(2);

    // Draw the triangle !
    //glDrawArrays(GL_TRIANGLES, 0, gl_draw_size); // number of vertices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangle_buffer);
    glDrawElements(GL_TRIANGLES, gl_draw_size, GL_UNSIGNED_INT, (void*)0);

    // (optional) unbind to avoid accidental modification of the buffers
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
}
#endif
