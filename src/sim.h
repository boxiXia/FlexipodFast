/*modified from the orginal Titan simulation libaray:https://github.com/jacobaustin123/Titan
ref: J. Austin, R. Corrales-Fatou, S. Wyetzner, and H. Lipson, �Titan: A Parallel Asynchronous Library for Multi-Agent and Soft-Body Robotics using NVIDIA CUDA,� ICRA 2020, May 2020.

Notes:
__CUDACC__ defines whether nvcc is steering compilation or not
__CUDA_ARCH__is always undefined when compiling host code, steered by nvcc or not
__CUDA_ARCH__is only defined for the device code trajectory of compilation steered by nvcc
For external library that works with cpp but not with .cu, wrap host code with
#ifndef __CUDA_ARCH__
....
#endif

*/

#ifndef TITAN_SIM_H
#define TITAN_SIM_H

#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#endif // !CUDA_API_PER_THREAD_DEFAULT_STREAM

#include "object.h"
#include "vec.h"
#include "comonUtils.h"

#include <msgpack.hpp>

#ifdef GRAPHICS
#include "shader.h"

#include <GL/glew.h>// Include GLEW
#include <GLFW/glfw3.h>// Include GLFW
#include <glm/glm.hpp>// Include GLM
#include <glm/gtc/matrix_transform.hpp>
#include<glm/gtx/rotate_vector.hpp>
#include<glm/gtx/norm.hpp>
#include <cuda_gl_interop.h>
#endif

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
//#include <cooperative_groups.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <exception>
#include <algorithm>
#include <list>
#include <vector>
#include <deque>
#include <set>
#include <numeric>      // std::accumulate

#include <thread>
#include <mutex>
#include <condition_variable>

#include <sstream>
#include <fstream>
#include <iostream>
#include <string>

#define _USE_MATH_DEFINES
#include <math.h>



class Simulation;

constexpr const int NUM_CUDA_STREAM = 4; // number of cuda stream excluding the default stream
constexpr const int CUDA_DYNAMICS_STREAM = 0;  // stream to run the dynamics update
constexpr const int CUDA_MEMORY_STREAM = 1;  // stream to run the memory operations
constexpr const int CUDA_MEMORY_STREAM_ALT = 2;  // additional stream to run the memory operations
constexpr const int CUDA_GRAPHICS_STREAM = 3; // steam to run graphics update

constexpr int MAX_BLOCKS = 65535; // max number of CUDA blocks

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "Cuda failure: %s %s %d\n", cudaGetErrorString(code), file, line);
		exit(code);
	}
}

#ifdef GRAPHICS
GLenum glCheckError_(const char* file, int line);
#define glCheckError() glCheckError_(__FILE__, __LINE__) //helper function to check OpenGL error
#endif


/*  helper function to free device/host memory given host_or_device_ptr */
using cudaFreeFcnType = cudaError_t(*)(void*); // helper type for the free memory function
inline cudaFreeFcnType FreeMemoryFcn(void* host_or_device_ptr) {
	cudaFreeFcnType freeMemory;
	cudaPointerAttributes attributes;
	gpuErrchk(cudaPointerGetAttributes(&attributes, host_or_device_ptr));
	if (attributes.type == cudaMemoryType::cudaMemoryTypeHost) {// host memory
		freeMemory = &cudaFreeHost;
	}
	else { freeMemory = &cudaFree; }// device memory
	printf("Memory type for d_data %i\n", attributes.type);
	return freeMemory;
}

template<class T>
bool isHostPointer(T* host_or_device_ptr) {
	cudaPointerAttributes attributes;
	cudaPointerGetAttributes(&attributes, (void*)host_or_device_ptr);
	return bool(attributes.type == cudaMemoryTypeHost);
}

/*  helper function to free device/host memory
	choose the appropriate device/host free memory function given a array pointer ptr */
template<class T>
cudaError_t __stdcall freeMemory(const bool on_host, T* ptr) {
	// if allocateMemory = cudaMallocHost: allocate on host
	if (on_host) { return cudaFreeHost(ptr); }
	else { return cudaFree(ptr); }
}


/*  helper function to allocate device/host memory
	choose the appropriate device/host free memory function given a array pointer ptr
	and its size (number)*/
template<class T>
cudaError_t __stdcall allocateMemory(const bool on_host, const size_t size, T*& ptr) {
	// if allocateMemory = cudaMallocHost: allocate on host
	if (on_host) { return cudaMallocHost((void**)&ptr, size * sizeof(T)); }
	else { return cudaMalloc((void**)&ptr, size * sizeof(T)); }
}

/* helper function to set host/device memory
Args:
	ptr: pointer to host/device memory
	value: Value (converted to usigned char) to set for each byte of specified memory
	size: number of <T> in the array
	on_host: if true, set host memory, else set device memory */
template<class T>
void __stdcall setMemory(T* ptr, int value, const size_t size, const bool on_host) {
	if (on_host) { memset(ptr, value, size * sizeof(T)); }
	else { cudaMemset(ptr, value, size * sizeof(T)); }
}


struct StdJoint {
	std::vector<int> left;// the indices of the left points
	std::vector<int> right;// the indices of the right points
	std::vector<int> anchor;// the 2 indices of the anchor points: left_anchor_id,right_anchor_id
	int left_coord;
	int right_coord;
	Vec3d axis;
#ifndef __CUDACC__ // not defined when compiling host code
	MSGPACK_DEFINE_MAP(left, right, anchor, left_coord, right_coord, axis);
#endif
};

class Model {
public:
	double radius_poisson = 0;// poisson discretization radius
	std::vector<Vec3d> vertices;// the mass xyzs
	std::vector<Vec2i> edges;//the spring ids
	std::vector<Vec3i> triangles; // the triangle indices
	std::vector<bool> is_surface;// whether the mass is near the surface
	std::map < std::string, std::vector<int>> id_vertices;// the edge id of the vertices
	std::map < std::string, std::vector<int>> id_edges;// the edge id of the springs
	std::vector<Vec3d> colors;// the mass xyzs
	std::vector<StdJoint> joints;// the joints
	std::vector<int> id_selected_edges; // # selected edges for spring strain
#ifndef __CUDACC__
	MSGPACK_DEFINE_MAP(radius_poisson, vertices, edges, triangles, is_surface,
		id_vertices, id_edges, colors, joints, id_selected_edges); // write the member variables that you want to pack
#endif
	Model() {}
	Model(const std::string& file_path, bool versbose = true);
};


__constant__ constexpr int MASS_FLAG_DOF_X_ID = 0;
__constant__ constexpr int MASS_FLAG_DOF_Y_ID = 1;
__constant__ constexpr int MASS_FLAG_DOF_Z_ID = 2;
//whether to apply constrain on the mass, must be set true for constraint to work
__constant__ constexpr int MASS_FLAG_CONSTRAIN_ID = 3;

// constexpr uint8_t MASS_FLAG_DOF       = 0b00000111;
__constant__ constexpr uint8_t MASS_FLAG_DOF = (1U << MASS_FLAG_DOF_X_ID) | (1U << MASS_FLAG_DOF_Y_ID) | (1U << MASS_FLAG_DOF_Z_ID);
// constexpr uint8_t MASS_FLAG_CONSTRAIN = 0b00001000;
__constant__ constexpr uint8_t MASS_FLAG_CONSTRAIN = (1U << MASS_FLAG_CONSTRAIN_ID);

struct MASS {
	//double* m = nullptr;
	double* inv_m = nullptr; // inverse mass
	Vec3d* pos = nullptr;
	Vec3d* pos_prev = nullptr;
	Vec3d* vel = nullptr;
	Vec3d* acc = nullptr;
	Vec3d* force = nullptr; // (measured) total force
	Vec3d* force_extern = nullptr; // (input) external force
	Vec3d* force_constraint = nullptr;// (measured) constrain force
	Vec3d* color = nullptr;
	Vec8b* flag = nullptr; // flag:(0-2:dof,3 constrain)

	int num = 0;
	inline int size() const { return num; }

	MASS() { }
	MASS(int num, bool on_host) {
		init(num, on_host);
	}
	/* initialize and copy the state from other MASS object. must keep the second argument*/
	MASS(MASS other, bool on_host, cudaStream_t stream = (cudaStream_t)0) {
		init(other.num, on_host);
		copyFrom(other, stream);
	}
	void init(int num, bool on_host = true) {
		//allocateMemory(on_host, num, m);
		allocateMemory(on_host, num, inv_m);
		allocateMemory(on_host, num, pos);
		allocateMemory(on_host, num, pos_prev);
		allocateMemory(on_host, num, vel);
		allocateMemory(on_host, num, acc);
		allocateMemory(on_host, num, force);
		allocateMemory(on_host, num, force_extern);
		allocateMemory(on_host, num, force_constraint);
		allocateMemory(on_host, num, color);
		allocateMemory(on_host, num, flag);
		gpuErrchk(cudaPeekAtLastError());
		this->num = num;
		// set inital values
		setMemory(vel, 0, num, on_host);
		setMemory(acc, 0, num, on_host);
		setMemory(flag, MASS_FLAG_DOF, num, on_host);// set dof flag
	}

	//~MASS() {
	//	std::cout << "****memory free****\n";
	//	//bool on_host = isHostPointer(pos);
	//	//freeMemory(on_host, inv_m);
	//	//freeMemory(on_host, pos);
	//	//freeMemory(on_host, pos_prev);
	//	//freeMemory(on_host, vel);
	//	//freeMemory(on_host, acc);
	//	//freeMemory(on_host, force);
	//	//freeMemory(on_host, force_extern);
	//	//freeMemory(on_host, force_constraint);
	//	//freeMemory(on_host, color);
	//	//freeMemory(on_host, flag);
	//}

	void copyFrom(const MASS& other, cudaStream_t stream = (cudaStream_t)0) {
		//cudaMemcpyAsync(m, other.m, num * sizeof(double), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(inv_m, other.inv_m, num * sizeof(double), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(pos, other.pos, num * sizeof(Vec3d), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(pos_prev, other.pos_prev, num * sizeof(Vec3d), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(vel, other.vel, num * sizeof(Vec3d), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(acc, other.acc, num * sizeof(Vec3d), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(force, other.force, num * sizeof(Vec3d), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(force_extern, other.force_extern, num * sizeof(Vec3d), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(force_constraint, other.force_constraint, num * sizeof(Vec3d), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(color, other.color, num * sizeof(Vec3d), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(flag, other.flag, num * sizeof(Vec8b), cudaMemcpyDefault, stream);
		//this->num = other.num;
		gpuErrchk(cudaPeekAtLastError());
	}
	void CopyPosVelAccFrom(MASS& other, cudaStream_t stream = (cudaStream_t)0) {
		cudaMemcpyAsync(pos, other.pos, num * sizeof(Vec3d), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(vel, other.vel, num * sizeof(Vec3d), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(acc, other.acc, num * sizeof(Vec3d), cudaMemcpyDefault, stream);
		//gpuErrchk(cudaPeekAtLastError());
	}
	void CopyPosFrom(MASS& other, cudaStream_t stream = (cudaStream_t)0) {
		cudaMemcpyAsync(pos, other.pos, num * sizeof(Vec3d), cudaMemcpyDefault, stream);
	}
	void CopyPosFrom(MASS& other, const int& index, const int& range = 1, cudaStream_t stream = (cudaStream_t)0) {
		assert(index + range <= num);//"index out of range"
		cudaMemcpyAsync(pos + index * sizeof(Vec3d), other.pos + index * sizeof(Vec3d), sizeof(Vec3d) * range, cudaMemcpyDefault, stream);
	}
	void CopyConstraintForceFrom(MASS& other, cudaStream_t stream = (cudaStream_t)0) {
		cudaMemcpyAsync(force_constraint, other.force_constraint, num * sizeof(Vec3d), cudaMemcpyDefault, stream);
	}
};

struct TERRAININFO {
	Vec3d* pos;
	Vec3d* normal;

	// here num represents the normalized diameter of the plane
	int num = 0;
	inline int size() const { return num; }
	int posnum = 0;
	int normalnum = 0;

	TERRAININFO() {}
	TERRAININFO(int num, bool on_host) {
		init(num, on_host);
	}
	// initialize and copy the state from other SPRING object. must keep the second argument//
	TERRAININFO(TERRAININFO other, bool on_host, cudaStream_t stream = (cudaStream_t)0) {
		init(other.num, on_host);
		copyFrom(other, stream);
	}

	void init(int num, bool on_host = true) {
		this->num = num;
		this->posnum = (num + 1) * (num + 1);
		this->normalnum = num * num * 2;

		allocateMemory(on_host, this->posnum, pos);
		allocateMemory(on_host, this->normalnum, normal);
		gpuErrchk(cudaPeekAtLastError());
	}

	void copyFrom(const TERRAININFO& other, cudaStream_t stream = (cudaStream_t)0) {
		cudaMemcpyAsync(pos, other.pos, posnum * sizeof(Vec3d), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(normal, other.normal, normalnum * sizeof(Vec3d), cudaMemcpyDefault, stream);
		gpuErrchk(cudaPeekAtLastError());
	}

	void copyFrom(const std::vector<Vec3d> posdata, const std::vector<Vec3d> normaldata, cudaStream_t stream = (cudaStream_t)0) {
		cudaMemcpyAsync(pos, &posdata[0], posnum * sizeof(Vec3d), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(normal, &normaldata[0], normalnum * sizeof(Vec3d), cudaMemcpyDefault, stream);
		gpuErrchk(cudaPeekAtLastError());
	}
};

struct SPRING {
	double* compliance = nullptr; // spring constant (N/m)
	double* rest = nullptr; // spring rest length (meters)
	double* damping = nullptr; // damping on the masses.
	Vec2i* edge = nullptr;// (left,right) mass indices of the spring
	bool* resetable = nullptr; // a flag indicating whether to reset every dynamic update
	int* joint_id = nullptr; // pointer to joint spring id
	double* lambda = nullptr; // lagrange multiplier 
	int num = 0;
	inline int size() { return num; }

	SPRING() {}
	SPRING(int num, bool on_host) {
		init(num, on_host);
	}
	/* initialize and copy the state from other SPRING object. must keep the second argument*/
	SPRING(SPRING other, bool on_host, cudaStream_t stream = (cudaStream_t)0) {
		init(other.num, on_host);
		copyFrom(other, stream);
	}

	void init(int num, bool on_host = true) { // initialize
		allocateMemory(on_host, num, compliance);
		allocateMemory(on_host, num, rest);
		allocateMemory(on_host, num, damping);
		allocateMemory(on_host, num, edge);
		allocateMemory(on_host, num, resetable);
		allocateMemory(on_host, num, joint_id);
		allocateMemory(on_host, num, lambda);
		gpuErrchk(cudaPeekAtLastError());
		this->num = num;
	}
	void copyFrom(const SPRING& other, cudaStream_t stream = (cudaStream_t)0) { // assuming we have enough streams
		cudaMemcpyAsync(compliance, other.compliance, num * sizeof(double), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(rest, other.rest, num * sizeof(double), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(damping, other.damping, num * sizeof(double), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(edge, other.edge, num * sizeof(Vec2i), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(resetable, other.resetable, num * sizeof(bool), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(joint_id, other.joint_id, num * sizeof(int), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(lambda, other.lambda, num * sizeof(double), cudaMemcpyDefault, stream);
		gpuErrchk(cudaPeekAtLastError());
		//this->num = other.num;
	}

};

// for displaying triangle mesh
struct TRIANGLE {
	Vec3i* triangle = nullptr; // triangle indices
	int num = 0; // number of triangles
	inline int size() { return num; }
	TRIANGLE() {}
	TRIANGLE(int num, bool on_host) {
		init(num, on_host);
	}
	/* initialize and copy the state from other SPRING object. must keep the second argument*/
	TRIANGLE(TRIANGLE other, bool on_host, cudaStream_t stream = (cudaStream_t)0) {
		init(other.num, on_host);
		copyFrom(other, stream);
	}
	void init(int num, bool on_host = true) { // initialize
		allocateMemory(on_host, num, triangle);
		gpuErrchk(cudaPeekAtLastError());
		this->num = num;
	}
	void copyFrom(const TRIANGLE& other, cudaStream_t stream = (cudaStream_t)0) { // assuming we have enough streams
		cudaMemcpyAsync(triangle, other.triangle, num * sizeof(Vec3i), cudaMemcpyDefault, stream);
		gpuErrchk(cudaPeekAtLastError());
		//this->num = other.num;
	}
};

/**********************************/
template <typename T>
struct RollingBuffer {
	T* data;
	int idx = 0;
	int num = 0;
	RollingBuffer(int num, bool on_host = true) :num(num) { allocateMemory(on_host, num, data); }
	void add(T value) { data[idx] = value; idx = (idx + 1) % num; }
	void clear() { setMemory(data, 0, num, isHostPointer(data)); idx = 0; }
	inline int size() { return num; }
};
/**********************************/

// for storing grid info
struct GridInfo {
	// store two arrays
	std::vector<std::vector<int>> triangle_vector;
	int* triangle_grid;
	Vec3d min_constraints;
	Vec3d cell_dimension;
	int grid_radius;
	int grid_size;
	int triangle_per_cell;
	bool grid_init = false;

	void init(int triangle_num)
	{
		allocateMemory(false, triangle_num, triangle_grid);
		gpuErrchk(cudaPeekAtLastError());
		setMemory(triangle_grid, -1, triangle_num, false);
	}

	void resetMemory()
	{
		setMemory(triangle_grid, -1, grid_size * triangle_per_cell, false);
	}
};

struct JOINT {
	/*--------------- joint-----------------------*/
	int num; // num of joint
	Vec2i* anchor; // index of the (left,right) anchor of the joint
	double* theta;// the angular increment per joint update
	double* pos; // joint position [rad] (-pi,pi]
	double* pos_desired; // desired joint position [rad] (-pi,pi]
	double* vel_desired; // desired joint velocity [rad/s]
	int* left_coord; // the index of left coordintate (x,y,z,-x,-y,-z) start for all joints (flat view)
	int* right_coord;// the index of right coordintate (x,y,z,-x,-y,-z) start for all joints (flat view)
	double* torque; // computed torque
	/*-------------- vertices ---------------------*/
	int vert_num; // number of vertices
	int* vert_id; // mass index of the left mass and right mass
	int* vert_joint_id;// e.g, k: index of the joint the point belongs to
	int* vert_dir; // direction: left=-1,right=+1
	/*--------------- edges -----------------------*/
	int edge_num; // number of edges
	int* edge_id; // spring index of the left mass and right mass
	int* edge_joint_id; // e.g, k: index of the joint the edge belongs to
	Vec3d* edge_c; // edge constant (c0,c1,phase)

	// return num of joints
	inline int size() const { return num; }

	JOINT() {};
	JOINT(const Model& robot, bool on_host = true) { init(robot, on_host); };

	/* initialize and copy the state from other JOINT object. must keep the second argument*/
	JOINT(JOINT other, bool on_host, cudaStream_t stream = (cudaStream_t)0) {
		init(other.num, other.vert_num, other.edge_num, on_host);
		copyFrom(other, stream);
	}

	void init(int num, int vert_num, int edge_num, bool on_host) {
		// joint
		this->num = num;
		allocateMemory(on_host, num, anchor);
		allocateMemory(on_host, num, theta);
		allocateMemory(on_host, num, pos);
		allocateMemory(on_host, num, pos_desired);
		allocateMemory(on_host, num, vel_desired);
		allocateMemory(on_host, num, left_coord);
		allocateMemory(on_host, num, right_coord);
		allocateMemory(on_host, num, torque);
		// joint vertices
		this->vert_num = vert_num;
		allocateMemory(on_host, vert_num, vert_id);
		allocateMemory(on_host, vert_num, vert_joint_id);
		allocateMemory(on_host, vert_num, vert_dir);
		// joint edges (friction spring)
		this->edge_num = edge_num;
		allocateMemory(on_host, edge_num, edge_id);
		allocateMemory(on_host, edge_num, edge_joint_id);
		allocateMemory(on_host, edge_num, edge_c);
		gpuErrchk(cudaPeekAtLastError());

	}

	void copyFrom(const JOINT& other, cudaStream_t stream = (cudaStream_t)0) {
		// joint
		cudaMemcpyAsync(anchor, other.anchor, num * sizeof(Vec2i), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(theta, other.theta, num * sizeof(double), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(pos, other.pos, num * sizeof(double), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(pos_desired, other.pos_desired, num * sizeof(double), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(vel_desired, other.vel_desired, num * sizeof(double), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(left_coord, other.left_coord, num * sizeof(int), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(right_coord, other.right_coord, num * sizeof(int), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(torque, other.torque, num * sizeof(double), cudaMemcpyDefault, stream);
		// joint vertices
		cudaMemcpyAsync(vert_id, other.vert_id, vert_num * sizeof(int), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(vert_joint_id, other.vert_joint_id, vert_num * sizeof(int), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(vert_dir, other.vert_dir, vert_num * sizeof(int), cudaMemcpyDefault, stream);
		// joint edges
		cudaMemcpyAsync(edge_id, other.edge_id, edge_num * sizeof(int), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(edge_joint_id, other.edge_joint_id, edge_num * sizeof(int), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(edge_c, other.edge_c, edge_num * sizeof(Vec3d), cudaMemcpyDefault, stream);

		gpuErrchk(cudaPeekAtLastError());
	}

	void init(const Model& robot, bool on_host = true) {

		const auto& fri_spring = robot.id_edges.at("fri_spring");//edges ids
		const auto& mass_pos = robot.vertices;
		const auto& std_joints = robot.joints;
		const auto& edges = robot.edges;
		num = std_joints.size();
		vert_num = 0;
		for each (const auto & std_joint in std_joints)// get the total number of the points in all joints
			vert_num += std_joint.left.size() + std_joint.right.size();
		edge_num = fri_spring.back() - fri_spring.front();

		init(num, vert_num, edge_num, on_host);
		if (on_host) { // copy the std_joints to this
			int vert_offset = 0;//offset the vert index by vert_offset
			int edge_offset = 0;//offset the edge index by edge_offset

			for (int jid = 0; jid < num; jid++)// jid:joint_index
			{
				const StdJoint& std_joint = std_joints[jid];
				anchor[jid] = std_joint.anchor;
				left_coord[jid] = std_joint.left_coord;
				right_coord[jid] = std_joint.right_coord;

				Vec3d rotation_axis = (mass_pos[anchor[jid].y] -
					mass_pos[anchor[jid].x]).normalize();
				// 0  1  2  3  4  5
				// x  y  z -x -y -z
				Vec3d y_left = mass_pos[left_coord[jid] + 1] - mass_pos[left_coord[jid] + 4];//(x,y,z,-x,-y,-z)
				Vec3d y_right = mass_pos[right_coord[jid] + 1] - mass_pos[right_coord[jid] + 4];//(x,y,z,-x,-y,-z)
				pos[jid] = signedAngleBetween(y_left, y_right, rotation_axis); //joint angle in [-pi,pi]
				// pos_desired[jid] = pos[jid];
				/*----------- joint vertices ------------------*/
				auto flattenVert = [&](auto& vert, int dir) {
					// ref: https://docs.microsoft.com/en-us/cpp/cpp/lambda-expressions-in-cpp
					for (auto i = 0; i < vert.size(); i++)
					{
						vert_id[vert_offset + i] = vert[i];
						vert_joint_id[vert_offset + i] = jid;
						vert_dir[vert_offset + i] = dir;
					}
					vert_offset += vert.size();//increment offset by num of left/right
				};
				flattenVert(std_joint.left, -1);
				flattenVert(std_joint.right, 1);

				/*----------- joint fri_spring ----------------*/
				Vec3d p_anchor_0 = mass_pos[anchor[jid].x]; // anchor_0 vertex pos
				double joint_pos = pos[jid]; // joint position [rad]

				int fri_spring_start = fri_spring[jid]; // fri_spring index start (inclusive)
				int fri_spring_end = fri_spring[jid + 1]; // fri_spring index end (exclusive)
				int fri_len = fri_spring_end - fri_spring_start;
				for (auto i = 0; i < fri_len; i++)
				{
					Vec2i ve = edges[i + fri_spring_start]; //left,right id of this fri_spring
					// vector of v0 and v1
					Vec3d v0 = mass_pos[ve.x] - p_anchor_0;
					Vec3d v1 = mass_pos[ve.y] - p_anchor_0;

					// distance tangential to rotation_axis 
					double d0_t = v0.dot(rotation_axis);
					double d1_t = v1.dot(rotation_axis);
					double d01_t = d1_t - d0_t;

					// vector normal to rotation_axis
					Vec3d v0_n = v0 - d0_t * rotation_axis;
					Vec3d v1_n = v1 - d1_t * rotation_axis;

					// distance normal to rotation_axis
					double d0_n = v0_n.norm();
					double d1_n = v1_n.norm();

					// signed angle between v0_n and v1_n about rotation_axis
					double t01 = signedAngleBetween(v0_n, v1_n, rotation_axis);
					// phase offset
					double edge_phase = t01 - joint_pos; // joint_pos + edge_phase = t01


					edge_c[edge_offset + i] = Vec3d(
						d0_n * d0_n + d1_n * d1_n + d01_t * d01_t,
						-2 * d0_n * d1_n,
						edge_phase);

					edge_id[edge_offset + i] = i + fri_spring_start;
					edge_joint_id[edge_offset + i] = jid;
				}
				edge_offset += fri_len;
			}
		}
	}


	void copyThetaFrom(const JOINT& other, cudaStream_t stream = (cudaStream_t)0) {
		gpuErrchk(cudaMemcpyAsync(theta, other.theta, num * sizeof(double), cudaMemcpyDefault, stream));
	}

	void copyPosFrom(const JOINT& other, cudaStream_t stream = (cudaStream_t)0) {
		gpuErrchk(cudaMemcpyAsync(pos, other.pos, num * sizeof(double), cudaMemcpyDefault, stream));
	}
};

// enum for the JointControl mode
enum class JointControlMode {
	vel = 0, // velocity control mode
	pos = 1, // position control mode
};


struct JointControl {

	JointControlMode mode = JointControlMode::vel;

	double* pos; // (measured) joint angle array in rad
	double* vel; // (measured) joint speed array in rad/s

	double* pos_desired;// (desired) joint angle array in rad
	double* vel_desired; // (desired) joint speed array in rad/s

	double* vel_error; // difference between cmd and vel
	double* pos_error; // integral of the error between joint_vel_cm and vel

	double* cmd; // (commended) joint speed array in rad/s

	double max_vel; // [rad/s] maximum joint speed
	double max_acc; // [rad/s^2] maximum joint acceleration
	double k_vel = 1.0; // coefficient for speed control
	double k_pos = 0.1; // coefficient for position control

	int num = 0;
	inline int size() const { return num; }
	JointControl() {}
	JointControl(int num, bool on_host) {
		init(num, on_host);
	}
	/* initialize and copy the state from other SPRING object. must keep the second argument*/
	JointControl(JointControl other, bool on_host, cudaStream_t stream = (cudaStream_t)0) {
		init(other.num, on_host);
		copyFrom(other, stream);
	}

	void init(int num, bool on_host = true) { // initialize
		allocateMemory(on_host, num, pos);
		allocateMemory(on_host, num, vel);
		allocateMemory(on_host, num, pos_desired);
		allocateMemory(on_host, num, vel_desired);
		allocateMemory(on_host, num, vel_error);
		allocateMemory(on_host, num, pos_error);
		allocateMemory(on_host, num, cmd);
		gpuErrchk(cudaPeekAtLastError());
		this->num = num;
	}

	void copyFrom(const JointControl& other, cudaStream_t stream = (cudaStream_t)0) {
		cudaMemcpyAsync(pos, other.pos, num * sizeof(double), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(vel, other.vel, num * sizeof(double), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(pos_desired, other.pos_desired, num * sizeof(double), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(vel_desired, other.vel_desired, num * sizeof(double), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(vel_error, other.vel_error, num * sizeof(double), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(pos_error, other.pos_error, num * sizeof(double), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(cmd, other.cmd, num * sizeof(double), cudaMemcpyDefault, stream);
		gpuErrchk(cudaPeekAtLastError());
		// todo make max_vel etc a vector
	}

	void setMaxJointSpeed(double max_joint_vel) {
		max_vel = max_joint_vel;
	}
	void reset(const MASS& mass, const JOINT& joint) {
		size_t nbytes = joint.size() * sizeof(double);
		memset(cmd, 0, nbytes);//cmd[i] = 0.;
		memset(vel, 0, nbytes);//vel[i] = 0.;
		memset(vel_desired, 0, nbytes);//vel_desired[i] = 0.;
		memset(vel_error, 0, nbytes);//vel_error[i] = 0.;
		memset(pos_error, 0, nbytes);//pos_error[i] = 0.;
		for (int i = 0; i < num; i++) {//TODO change this...
			Vec2i anchor_edge = joint.anchor[i];
			Vec3d rotation_axis = (mass.pos[anchor_edge.y] - mass.pos[anchor_edge.x]).normalize();
			// 0  1  2  3  4  5
			// x  y  z -x -y -z
			Vec3d y_left = mass.pos[joint.left_coord[i] + 1] - mass.pos[joint.left_coord[i] + 4];//(x,y,z,-x,-y,-z)
			Vec3d y_right = mass.pos[joint.right_coord[i] + 1] - mass.pos[joint.right_coord[i] + 4];//(x,y,z,-x,-y,-z)
			pos[i] = signedAngleBetween(y_left, y_right, rotation_axis); //joint angle in [-pi,pi]
			pos_desired[i] = pos[i];//keep it still
		}
	}

	/*update the jointcontrol state, ndt is the delta time between jointcontrol update*/
	void update(const MASS& mass, const JOINT& joint, double ndt) {
		//#pragma omp simd
		for (int i = 0; i < num; i++) // compute joint angles and angular velocity
		{
			Vec2i anchor_edge = joint.anchor[i];
			Vec3d rotation_axis = (mass.pos[anchor_edge.y] - mass.pos[anchor_edge.x]).normalize();
			// 0  1  2  3  4  5
			// x  y  z -x -y -z
			Vec3d y_left = mass.pos[joint.left_coord[i] + 1] - mass.pos[joint.left_coord[i] + 4];//(x,y,z,-x,-y,-z)
			Vec3d y_right = mass.pos[joint.right_coord[i] + 1] - mass.pos[joint.right_coord[i] + 4];//(x,y,z,-x,-y,-z)
			// angle is within (-pi,pi]
			double angle = signedAngleBetween(y_left, y_right, rotation_axis); //joint angle in [-pi,pi]

			double delta_angle = angle - pos[i];
			// assuming delta_angle is within (-pi,pi)
			// there may be a huge chage from -pi to pi, but really it only moves a little bit
			clampPeroidicInplace(delta_angle, -M_PI, M_PI);
			vel[i] = delta_angle / ndt;

			pos[i] = angle;

			// update cmd with position PD control: clamp to (-max_vel,max_vel)
			double dv;
			double vel_desired_proxy;
			switch (mode)//switch controller mode
			{
			case JointControlMode::vel:
				clampInplace(vel_desired[i], -max_vel, max_vel);

				//vel_error[i] = vel_desired[i] - vel[i];
				//pos_desired[i] += vel_desired[i] * ndt;
				//pos_error[i] = pos_desired[i] - pos[i];
				//clampPeroidicInplace(pos_desired[i], -M_PI, M_PI);
				//clampPeroidicInplace(pos_error[i], -M_PI, M_PI);
				//cmd[i] = k_vel * vel_error[i] + k_pos/ndt * pos_error[i];

				//TODO FIX THIS
				vel_desired_proxy = vel_desired[i];
				dv = max_acc * ndt;
				clampInplace(vel_desired_proxy, joint.vel_desired[i] - dv, joint.vel_desired[i] + dv);
				// set joint velocity
				joint.vel_desired[i] = vel_desired_proxy;
				cmd[i] = vel_desired_proxy;

				vel_desired[i] = 0.99 * vel_desired[i];
				break;
			case JointControlMode::pos:
				clampPeroidicInplace(pos_desired[i], -M_PI, M_PI);
				pos_error[i] = pos_desired[i] - pos[i];
				clampPeroidicInplace(pos_error[i], -M_PI, M_PI);


				vel_desired_proxy = 10 * pos_error[i] - 0.1 * vel[i];

				dv = max_acc * ndt;
				//clampInplace(vel_desired_proxy, vel_desired[i] - dv, vel_desired[i] + dv);
				clampInplace(vel_desired_proxy, vel[i] - dv, vel[i] + dv);
				clampInplace(vel_desired_proxy, -max_vel, max_vel);
				joint.vel_desired[i] = vel_desired_proxy;
				cmd[i] = joint.vel_desired[i];

				//vel_desired_proxy = pos_error[i] / ndt/20; // reach the destination at 50 control step
				//dv = max_acc * ndt;

				//clampInplace(vel_desired_proxy, vel_desired[i] - dv, vel_desired[i] + dv);
				//clampInplace(vel_desired_proxy, -max_vel, max_vel);

				//vel_desired[i] = vel_desired_proxy;
				//joint.vel_desired[i] = vel_desired[i];
				//cmd[i] = vel_desired[i];

				//vel_desired_proxy = pos_error[i] / ndt; // reach the destination at 1 control step
				//dv = max_acc * ndt;
				//clampInplace(vel_desired_proxy, vel_desired[i] - dv, vel_desired[i] + dv);
				//clampInplace(vel_desired_proxy, -max_vel, max_vel);
				//// calcuate settling time based ob velocity change
				//double ndt_proxy = (vel_desired_proxy - vel_desired[i]) / max_acc;
				//vel_desired[i] = vel_desired_proxy;
				//ndt_proxy = abs(pos_error[i] / max_vel)*2.0;
				//if (ndt_proxy < ndt) { ndt_proxy = ndt; }
				//vel_error[i] = vel_desired[i] - vel[i];
				////cmd[i] = k_vel * vel_error[i] +k_pos / ndt * pos_error[i];
				//cmd[i] = k_pos / ndt_proxy * pos_error[i];

				break;
			}
			clampInplace(cmd[i], -max_vel, max_vel);
		}
	}

	void updateControlMode(const JointControlMode& control_mode) {
		if (mode != control_mode) {
			mode = control_mode; // update mode to control_mode
			switch (mode)
			{
			case JointControlMode::vel: // changed to velocity control
				for (int i = 0; i < num; i++) {
					vel_desired[i] = 0;
					//vel_desired[i] = vel[i];
				}
				break;
			case JointControlMode::pos:// changed to postion control
				//for (int i = 0; i < num; i++) {
				//	pos_desired[i] = pos[i];
				//}
				break;
			}
		}
	}

};

/// <summary>
/// a rigidbody representation of attached coordinate (body) frame
/// </summary>
struct RigidBody {
	Vec3d pos; // position of the coordinate frame origin
	Vec3d vel; // velcotiy of the coordinate frame origin
	Vec3d acc; // acceleration of the coordinate frame origin
	//Vec3d ux; // x unit vector of the body frame
	//Vec3d uy; // y unit vector of the body frame
	//Vec3d uz; // z unit vector of the body frame
	Mat3d rot;// rotation matrix = [ux,uy,uz] of the frame
	Vec3d ang_vel;//angular velocity in body space
	int id_start; // start mass index of the coordinate frame

	MSGPACK_DEFINE_ARRAY(pos, vel, acc, rot, ang_vel);
	/// <summary>
	/// initialize the rigidbody, assuming the coordinate frame is at index id_start
	/// </summary>
	/// <param name="mass"> MASS struct storing the coordinate frame of the rigidbody</param>
	/// <param name="id_start"> start index of the coordinate frame</param>
	/// <param name="w"> initial angular velocity in body frame</param>
	void init(const MASS& mass, const int& id_start, Vec3d w = Vec3d(0, 0, 0)) {
		// com
		// 0  1  2  3  4  5
		// x  y  z -x -y -z
		this->id_start = id_start;
		pos = std::accumulate(mass.pos + id_start, mass.pos + id_start + 6, Vec3d(0., 0., 0.)) / 6.;
		vel = std::accumulate(mass.vel + id_start, mass.vel + id_start + 6, Vec3d(0., 0., 0.)) / 6.;
		acc = std::accumulate(mass.acc + id_start, mass.acc + id_start + 6, Vec3d(0., 0., 0.)) / 6.;
		Vec3d ux = (mass.pos[id_start] - mass.pos[id_start + 3]).normalize();
		Vec3d uy = (mass.pos[id_start + 1] - mass.pos[id_start + 4]);
		uy = (uy - uy.dot(ux) * ux).normalize();
		Vec3d uz = cross(ux, uy);
		rot = Mat3d(ux, uy, uz, false);
		this->ang_vel = w;
	}
	void update(const MASS& mass, const double& dt) {
		// instead of directly estimating the acc,
		// estimate from vel backward difference

		Vec3d pos_new = std::accumulate(mass.pos + id_start, mass.pos + id_start + 6, Vec3d(0., 0., 0.)) / 6.;

		//Vec3d pos_new = mass.pos[id_start];
		Vec3d vel_new = (pos_new - pos) / dt;
		acc = (vel_new - vel) / dt;
		vel = vel_new;
		//pos = mass.pos[id_start];
		pos = pos_new;
		// 0  1  2  3  4  5
		// x  y  z -x -y -z
		Vec3d ux = (mass.pos[id_start] - mass.pos[id_start + 3]).normalize();
		Vec3d uy = (mass.pos[id_start + 1] - mass.pos[id_start + 4]);
		uy = (uy - uy.dot(ux) * ux).normalize();
		Vec3d uz = cross(ux, uy).normalize();
		Mat3d rot_new = Mat3d(ux, uy, uz, false);

		ang_vel = Mat3d::angularVelocityFromRotation(rot, rot_new, dt, true);

		rot = rot_new;
	}

	char* print() {
		char out[300];
		int n = snprintf(out, 300,
			"com pos %+6.2f %+6.2f %+6.2f |%6.2f|\n"
			"com vel %+6.2f %+6.2f %+6.2f |%6.2f|\n"
			"com acc %+6.2f %+6.2f %+6.2f |%6.2f|\n"
			"ang vel %+6.2f %+6.2f %+6.2f |%6.2f|\n"
			"rotation:\n"
			"%+7.3f %+7.3f %+7.3f\n"
			"%+7.3f %+7.3f %+7.3f\n"
			"%+7.3f %+7.3f %+7.3f\n",
			pos.x, pos.y, pos.z,pos.norm(),
			vel.x, vel.y, vel.z,vel.norm(),
			acc.x, acc.y, acc.z,acc.norm(),
			ang_vel.x, ang_vel.y, ang_vel.z, ang_vel.norm(),
			rot.m00, rot.m01, rot.m02,
			rot.m10, rot.m11, rot.m12,
			rot.m20, rot.m21, rot.m22
		);
		//assert(n < 250);
		//printf("%d\n",n);
		return out;
	}

};


struct BreakPoint {
	double t;
	bool should_end;
	BreakPoint(double t = 0, bool should_end = false) :t(t), should_end(should_end) {}
	constexpr bool operator() (BreakPoint const& p1, BreakPoint const& p2) const {
		return p1.t < p2.t;
	}
};



#ifdef UDP
enum UDP_HEADER :int {
	TERMINATE = -1,// close the program
	PAUSE = 17,
	RESUME = 16,
	RESET = 15,
	ROBOT_STATE_REPORT = 14,
	MOTOR_SPEED_COMMEND = 13,
	STEP_MOTOR_SPEED_COMMEND = 12,
	MOTOR_POS_COMMEND = 11,
	STEP_MOTOR_POS_COMMEND = 10,
};
MSGPACK_ADD_ENUM(UDP_HEADER); // msgpack macro,refer to https://github.com/msgpack/msgpack-c/blob/cpp_master/example/cpp03/enum.cpp



class DataSend {/*the info to be sent to the high level controller*/
public:
	UDP_HEADER header = UDP_HEADER::ROBOT_STATE_REPORT;
	float T = 0; // time at the sending of this udp packet
	// using float to make data smaller
	std::vector<float> joint_pos; // joint position cos(angle),sin(angle) {-1,1}
	std::vector<float> joint_vel; // joint angular velocity [rad/s],{-1,1}
	std::vector<float> joint_torque; // acutation of the joint,{-1,1}
	//std::vector<float> joint_cmd; // action commanded by the remote controller
	float orientation[6] = { 0 }; // orientation of the body,{-1,1}
	float ang_vel[3];
	float com_acc[3];
	float com_vel[3];
	float com_pos[3];
#ifdef MEASURE_CONSTRAINT
	std::vector<float> constraint_force; // acutation of the joint,{-1,1}
#endif //MEASURE_CONSTRAINT
#ifdef STRESS_TEST
	std::vector<float> spring_strain;
#endif //STRESS_TEST
	//MSGPACK_DEFINE_ARRAY(header, T, joint_pos, joint_vel, joint_torque, orientation, ang_vel, com_acc, com_vel, com_pos, constraint_force, spring_strain);
	template <typename Packer>
	void msgpack_pack(Packer& msgpack_pk) const {
		msgpack::type::make_define_array(
			header, 
			T, 
			joint_pos, 
			joint_vel, 
			joint_torque, 
			//joint_cmd, 
			orientation, 
			ang_vel, 
			com_acc, 
			com_vel, 
			com_pos
#ifdef MEASURE_CONSTRAINT
			, constraint_force
#endif
#ifdef STRESS_TEST
			, spring_strain
#endif
		).msgpack_pack(msgpack_pk);
	}

	DataSend() {}// defualt constructor

	DataSend(
		const UDP_HEADER& header,
		const Simulation* s);

};

class DataReceive {/*the high level command to be received */
public:
	UDP_HEADER header;
	double T;
	std::vector<double> joint_value_desired;// desired joint position or velocity
	MSGPACK_DEFINE_ARRAY(header, T, joint_value_desired);

	DataReceive(int num_joint) {
		init(num_joint);
	}
	DataReceive() {}// defualt constructor
	void init(int num_joint) {
		joint_value_desired = std::vector<double>(num_joint, 0);
	}
};

//typedef WsaUdpServer< DataReceive, std::deque<DataSend>> UdpServer;


/// <summary>
/// 
/// </summary>
class asioUdpServer
{
public:

	std::string ip_local; uint16_t port_local;
	std::string ip_remote; uint16_t port_remote;

	bool UDP_SHOULD_RUN = true;// flag indicating whether to stop sending/receiving
	bool flag_new_received = false; // flag indicating a new message has received
	int counter_rec = 0; //  a counter for counting received message

	std::thread thread_send; // thread for sending udp
	std::thread thread_recv; // thread for receiving udp

	std::deque<std::string> msg_send_queue; // queue of data to be sent, new messages added to the back
	std::deque<std::string> msg_recv_queue; // queue of data received, new messages added to the back

	asioUdpServer() {};

	asioUdpServer(std::string ip_local, uint16_t port_local,
		std::string ip_remote, uint16_t port_remote) :
		ip_local(ip_local), port_local(port_local), ip_remote(ip_remote), port_remote(port_remote) {}

	/*run this to start receiving and sending udp*/
	void run() {
		UDP_SHOULD_RUN = true;
		thread_recv = std::thread([=]() { this->_doReceive(); });
		thread_send = std::thread([=]() { this->_doSend(); });
	}
	void close() {
		UDP_SHOULD_RUN = false;
		_joinThread();
	}

	~asioUdpServer() { _joinThread(); }

private:
	void _doReceive();
	void _doSend();

	inline void _joinThread() {
		if (thread_send.joinable()) { thread_send.join(); }
		if (thread_recv.joinable()) { thread_recv.join(); }
	}
};
#endif // UDP


class Simulation {
public:
	int device;

	double dt = 0.0001;
	double T = 0; //simulation time
	Vec3d global_acc = Vec3d(0, 0, 0); // global acceleration

	int id_oxyz_start = 0;// coordinate (x,y,z,-x,-y,-z) start index (inclusive)
	int id_oxyz_end = 0; //  coordinate (x,y,z,-x,-y,-z) end index (exclusive)

	// cuda and udp update parameters (should be constant during the simualtion)
	int NUM_QUEUED_KERNELS = 50; // number of kernels to queue at a given time (this will reduce the frequency of updates from the CPU by this factor
#ifdef UDP
	int udp_num_obs = 5;// send udp_num_obs at once (number of observations)
	int udp_step = 4; // udp observations is stepped by this factor
	int udp_delay_step = 0; // udp obervation is delayed by this step
	// combination that works:NUM_QUEUED_KERNELS = 50,udp_step=4,udp_num_obs=5
#endif
	// host
	MASS mass; // a flat fiew of all masses
	SPRING spring; // a flat fiew of all springs
	TRIANGLE triangle; // a flat view of all triangles
	JOINT joint;// a flat view of all joints
	JointControl joint_control; // joint controller
	RigidBody body; // mainbody
	// device
	MASS d_mass;
	SPRING d_spring;
	TRIANGLE d_triangle; // a flat view of all triangles
	JOINT d_joint;

	// 3d grid to store the grids containing which triangle
	GridInfo gridInfo;
	void updateGridInfo();
	void computeHash(double& min, double& min_coord, double& max_coord, double& radius, const int grid_radius, std::pair<int, int>& ret);

	// test for support terrains Mingxuan
	// Vec3d* d_vertices;
	// Vec3d* d_normals;
	TERRAININFO vertex;
	TERRAININFO d_vertex;

	// host (backup);
	MASS backup_mass;
	SPRING backup_spring;
	JOINT backup_joint;


#ifdef STRESS_TEST
	std::vector<int> id_selected_edges;
#endif //STRESS_TEST

#ifdef MEASURE_CONSTRAINT
	Vec3d force_constraint;
	float fc_max = 0; // maximum force constraint
	RollingBuffer<Vec3d> fc_arr{ 1024 };

	std::map < std::string, std::vector<int>> id_vertices;// the edge id of the vertices
	int num_body = 0; // number of body, should be assigned when initializing
	std::vector<Vec3d> body_constraint_force;

	float total_mass = 0;// total mass weight in [kg]
#endif //MEASURE_CONSTRAINT

	void backupState();//backup the robot mass/spring/joint state
	void resetState();// restore the robot mass/spring/joint state to the backedup state


	// state report
	bool RUNNING = false;
	bool STARTED = false;
	bool ENDED = false; // flag is set true when ~Simulation() is called

	// control
	bool RESET = false;// reset flag
	bool SHOULD_RUN = true;
	bool SHOULD_END = false;

	bool USE_PBD = true;// flag to use position based dynamics

	std::mutex mutex_running;
	std::condition_variable cv_running;

	cudaStream_t stream[NUM_CUDA_STREAM]; // cuda stream:https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution

	//Simulation();
	Simulation(size_t num_mass, size_t num_spring, size_t num_joint, size_t num_triangle, int device = 0);
	~Simulation();

	void getAll();
	void setAll();
	// Global constraints (can be rendered)
	// creates half-space ax + by + cz < d
	void createPlane(const Vec3d& abc, const double d, const double FRICTION_K = 0, const double FRICTION_S = 0,
		/*rendering:*/float square_size = 0.5f, float plane_radius = 5);
	void createBall(const Vec3d& center, const double r); // creates ball with radius r at position center
	// creates terrain that has randomized normals over the surface
	void createTerrain(const double FRICTION_K = 0, const double FRICTION_S = 0, float unit_size = 0.5f,
		float terrain_radius = 5, double terrain_waviness = 0.3);
	void clearConstraints(); // clears global constraints only

	void setBreakpoint(const double time, const bool should_end = false); // tell the program to stop at the set time (doesn't hang).
	void start();
	void pause(const double t = 0);//pause the simulation at (simulation) time t [s]
	void resume();

	void updatePhysics();
	void updateGraphics();

#ifdef DEBUG_ENERGY
	double energy(); //compute the total energy of the system
	double energy_start;
	double energy_deviation_max = 0;
#endif // DEBUG_ENERGY


#ifdef UDP
public:
	//                                       ip_local,port_local,ip_remote,port_remote
	asioUdpServer udp_server = asioUdpServer("127.0.0.1", 33301, "127.0.0.1", 33300);

	bool updateUdpReceive();
	void updateUdpSend();
	bool SHOULD_SEND_UDP = false; // update and send udp
	bool UDP_INIT = true; // bool to inform the udp thread to initialize

	std::thread thread_msg_update; // update udp message;
#endif //UDP

private:
	void waitForEvent();
	void freeGPU();
	void updateCudaParameters();
	int computeGridSize(int block_size, int num);//helper function to compute blocks per grid

	std::thread thread_physics_update;
#ifdef GRAPHICS
	std::thread thread_graphics_update;
	bool GRAPHICS_ENDED = false; // a flag set by thread_graphics_update to notify its termination
#endif //GRAPHICS

	std::set<BreakPoint, BreakPoint> bpts; // list of breakpoints

	int spring_block_size = 64; // spring update threads per block
	int spring_grid_size;// spring update blocks per grid

	int mass_block_size = 64; // mass update threads per block
	int mass_grid_size; // mass update blocks per grid

	int joint_block_size = 64; // joint rotate threads per blcok
	int joint_grid_size;// joint rotate blocks per grid

	int joint_edge_block_size = 64; //joint edge
	int joint_edge_grid_size;

#ifdef GRAPHICS
	int triangle_block_size = 64; // triangle update threads per block
	int triangle_grid_size; // triangle update blocks per grid

	int vertex_block_size = 64;// vertex update threads per block
	int vertex_grid_size; // vertex update blocks per grid
#endif //GRAPHICS

	std::vector<Constraint*> constraints;
	thrust::device_vector<CudaContactPlane> d_planes; // used for constraints
	thrust::device_vector<CudaBall> d_balls; // used for constraints
	thrust::device_vector<CudaContactTerrain> d_terrains;

	CUDA_GLOBAL_CONSTRAINTS d_constraints;
	bool SHOULD_UPDATE_CONSTRAINT = true; // a flag indicating whether constraint should be updated

#ifdef GRAPHICS
public:
	void setViewport(const glm::vec3& camera_position, const glm::vec3& target_location, const glm::vec3& up_vector);
	void moveViewport(const glm::vec3& displacement);

	glm::vec4 clear_color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f); // rgba, clearing window color when calling glClearColor 
	int contex_version_major = 4; // for GLFW_CONTEXT_VERSION_MAJOR
	int contex_version_minor = 6; // for GLFW_CONTEXT_VERSION_MINOR


	GLFWwindow* window;
	int framebuffer_width, framebuffer_height; // the width and height of the framebuffer

	Shader shader; // shader object, its handel is shader.ID, call shader.use() to use
	GLuint VertexArrayID; // handle for the vertex array object

	/*--------------shadow----------------------------*/
	Shader simpleDepthShader;
	GLuint depthMapFBO; // depth map frame buffer object
	GLuint depthMap; // handle for the depthmap texture
	const GLuint SHADOW_WIDTH = 1080;
	const GLuint SHADOW_HEIGHT = 1080;
	/*------------------------------------------------*/

	DirectionLight light; // directional light

	glm::mat4 model_matrix;// model_matrix matrix
	glm::mat4 view_matrix; // view matrix
	glm::mat4 projection_matrix; // projection matrix

	Camera camera;
	glm::vec3 camera_target_offset{0,0,0};


	void computeMVP(bool update_view = true); // compute MVP

	// imgui
	bool show_imgui = true; // show imgui window

	// function pointer to process additional callback
	void (*keyboardCallback)(Simulation*) = nullptr;

	// glfwGetKey for additional keyboard processing
	int getKey(int key) { return glfwGetKey(window, key); }


private:
	/*--------------------------------- ImGui ----------------------------------------*/
	void startupImgui();
	void runImgui();
	void shutdownImgui();

	/*------------- vertex buffer object and their device pointers--------------------*/
	GLuint vbo_vertex; // handle for vertexbuffer: for vertex position,color, vertex normal update
	VERTEX_DATA* dptr_vertex = nullptr;// used in updateBuffers(), device pointer,stores positions of the vertices
	struct cudaGraphicsResource* cuda_resource_vertex;

	GLuint vbo_edge; // handle for elementbuffer (spring)
	uint2* dptr_edge = nullptr; // used in updateBuffers(), device pointer,stores indices (line plot)
	struct cudaGraphicsResource* cuda_resource_edge;

	GLuint vbo_triangle; // handle for elementbuffer (spring)
	uint3* dptr_triangle = nullptr; // used in updateBuffers(), device pointer,stores indices (line plot)
	struct cudaGraphicsResource* cuda_resource_triangle;

	bool show_triangle = true;
	bool resize_buffers = true; // update all (vbo_vertex,vbo_color,vbo_edge, vbo_triangles) if true

	void updateBuffers();
	void updateVertexBuffers();//only update vertex (positions)
	void generateBuffers();
	void resizeBuffers();
	void deleteBuffers();

	void createVBO(GLuint* vbo, struct cudaGraphicsResource** vbo_res, size_t size, unsigned int vbo_res_flags, GLenum buffer_type = GL_ARRAY_BUFFER);
	void deleteVBO(GLuint* vbo, struct cudaGraphicsResource* vbo_res, GLenum buffer_type);
	void resizeVBO(GLuint* vbo, size_t size, GLenum buffer_type = GL_ARRAY_BUFFER);
	/*-------------------------------------------------------------------------------*/
	void draw();
	void createGLFWWindow();
	static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
	static void framebuffer_size_callback(GLFWwindow* window, int width, int height);

#endif
};

#ifdef GRAPHICS
#endif


#endif //TITAN_SIM_H