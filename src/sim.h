/*modified from the orginal Titan simulation libaray:https://github.com/jacobaustin123/Titan
ref: J. Austin, R. Corrales-Fatou, S. Wyetzner, and H. Lipson, �Titan: A Parallel Asynchronous Library for Multi-Agent and Soft-Body Robotics using NVIDIA CUDA,� ICRA 2020, May 2020.
*/

#ifndef TITAN_SIM_H
#define TITAN_SIM_H

#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#endif // !CUDA_API_PER_THREAD_DEFAULT_STREAM

#include "object.h"
#include "vec.h"
#include "shader.h"

#include <msgpack.hpp>

#ifdef GRAPHICS
#include "shader.h"

#include <GL/glew.h>// Include GLEW
#include <GLFW/glfw3.h>// Include GLFW
#include <glm/glm.hpp>// Include GLM
#include <glm/gtc/matrix_transform.hpp>
#include <cuda_gl_interop.h>
#endif

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <list>
#include <vector>
#include <set>

#include <thread>
#include <mutex>
#include <condition_variable>

#include <sstream>
#include <fstream>
#include<iostream>
#include<string.h>

#define _USE_MATH_DEFINES
#include <math.h>

#ifdef UDP
#include "Network.h"
typedef WsaUdpServer UdpServer;

#endif


constexpr const int NUM_CUDA_STREAM = 5; // number of cuda stream excluding the default stream
constexpr const int CUDA_DYNAMICS_STREAM = 0;  // stream to run the dynamics update
constexpr const int CUDA_MEMORY_STREAM = 1;  // stream to run the memory operations
constexpr const int CUDA_GRAPHICS_STREAM = 2; // steam to run graphics update

constexpr int  NUM_QUEUED_KERNELS = 40; // number of kernels to queue at a given time (this will reduce the frequency of updates from the CPU by this factor
constexpr int NUM_UPDATE_PER_ROTATION = 4; //number of update per rotation
constexpr int NUM_UDP_MULTIPLIER = 3;// udp update is decreased by this factor


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "Cuda failure: %s %s %d\n", cudaGetErrorString(code), file, line);
		exit(code);
	}
}





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

/*  helper function to allocate device/host memory
	choose the appropriate device/host free memory function given a array pointer, e.g:
	cudaFreeFcnType freeMemory = FreeMemoryFcn((void*)m);
	freeMemory((void*)m);*/
using cudaMallocFcnType = cudaError_t(*)(void**, size_t);
inline cudaMallocFcnType allocateMemoryFcn(bool on_host) {// ref: https://www.cprogramming.com/tutorial/function-pointers.html
	cudaMallocFcnType allocateMemory;
	if (on_host) { allocateMemory = &cudaMallocHost; }// if allocateMemory = cudaMallocHost: allocate on host
	else { allocateMemory = &cudaMalloc; }
	return allocateMemory;
}

struct StdJoint {
	std::vector<int> left;// the indices of the left points
	std::vector<int> right;// the indices of the right points
	std::vector<int> anchor;// the 2 indices of the anchor points: left_anchor_id,right_anchor_id
	int leftCoord;
	int rightCoord;
	Vec3d axis;
	MSGPACK_DEFINE(left, right, anchor, leftCoord, rightCoord, axis);
};
class Model {
public:
	double radius_poisson;// poisson discretization radius
	std::vector<Vec3d> vertices;// the mass xyzs
	std::vector<Vec2i> edges;//the spring ids
	std::vector<Vec3i> triangles; // the triangle indices
	std::vector<bool> isSurface;// whether the mass is near the surface
	std::vector<int> idVertices;// the edge id of the vertices
	std::vector<int> idEdges;// the edge id of the springs
	std::vector<Vec3d> colors;// the mass xyzs
	std::vector<StdJoint> Joints;// the joints
	MSGPACK_DEFINE(radius_poisson, vertices, edges,triangles, isSurface, idVertices, idEdges, colors, Joints) // write the member variables that you want to pack
		Model() {}
	Model(const char* file_path) {
		// get the msgpack robot model
		// Deserialize the serialized data
		std::ifstream ifs(file_path, std::ifstream::in | std::ifstream::binary);
		std::stringstream buffer;
		buffer << ifs.rdbuf();
		msgpack::unpacked upd;//unpacked data
		msgpack::unpack(upd, buffer.str().data(), buffer.str().size());
		//    std::cout << upd.get() << std::endl;
		upd.get().convert(*this);
	}
};



struct MASS {
	double* m = nullptr;
	Vec3d* pos = nullptr;
	Vec3d* vel = nullptr;
	Vec3d* acc = nullptr;
	Vec3d* force = nullptr;
	Vec3d* force_extern = nullptr;
	Vec3d* color = nullptr;
	bool* fixed = nullptr;
	bool* constrain = nullptr;//whether to apply constrain on the mass, must be set true for constraint to work
	int num = 0;
	inline int size() { return num; }

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
		cudaMallocFcnType allocateMemory = allocateMemoryFcn(on_host);// choose approipate malloc function
		allocateMemory((void**)&m, num * sizeof(double));
		allocateMemory((void**)&pos, num * sizeof(Vec3d));
		allocateMemory((void**)&vel, num * sizeof(Vec3d));
		allocateMemory((void**)&acc, num * sizeof(Vec3d));
		allocateMemory((void**)&force, num * sizeof(Vec3d));
		allocateMemory((void**)&force_extern, num * sizeof(Vec3d));
		allocateMemory((void**)&color, num * sizeof(Vec3d));
		allocateMemory((void**)&fixed, num * sizeof(bool));
		allocateMemory((void**)&constrain, num * sizeof(bool));

		gpuErrchk(cudaPeekAtLastError());
		this->num = num;
		if (on_host) {// set vel,acc to 0
			memset(vel, 0, num * sizeof(Vec3d));
			memset(acc, 0, num * sizeof(Vec3d));
		}
		else {
			cudaMemset(vel, 0, num * sizeof(Vec3d));
			cudaMemset(acc, 0, num * sizeof(Vec3d));
		}
	}

	void copyFrom(const MASS& other, cudaStream_t stream = (cudaStream_t)0) {
		cudaMemcpyAsync(m, other.m, num * sizeof(double), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(pos, other.pos, num * sizeof(Vec3d), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(vel, other.vel, num * sizeof(Vec3d), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(acc, other.acc, num * sizeof(Vec3d), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(force, other.force, num * sizeof(Vec3d), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(force_extern, other.force_extern, num * sizeof(Vec3d), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(color, other.color, num * sizeof(Vec3d), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(fixed, other.fixed, num * sizeof(bool), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(constrain, other.constrain, num * sizeof(bool), cudaMemcpyDefault, stream);
		//this->num = other.num;
		gpuErrchk(cudaPeekAtLastError());
	}
	void CopyPosVelAccFrom(MASS& other, cudaStream_t stream = (cudaStream_t)0) {
		cudaMemcpyAsync(pos, other.pos, num * sizeof(Vec3d), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(vel, other.vel, num * sizeof(Vec3d), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(acc, other.acc, num * sizeof(Vec3d), cudaMemcpyDefault, stream);
		gpuErrchk(cudaPeekAtLastError());
	}

};

struct SPRING {
	double* k = nullptr; // spring constant (N/m)
	double* rest = nullptr; // spring rest length (meters)
	double* damping = nullptr; // damping on the masses.
	Vec2i* edge = nullptr;// (left,right) mass indices of the spring
	bool* resetable = nullptr; // a flag indicating whether to reset every dynamic update
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
		cudaMallocFcnType allocateMemory = allocateMemoryFcn(on_host);// choose approipate malloc function
		allocateMemory((void**)&k, num * sizeof(double));
		allocateMemory((void**)&rest, num * sizeof(double));
		allocateMemory((void**)&damping, num * sizeof(double));
		allocateMemory((void**)&edge, num * sizeof(Vec2i));
		allocateMemory((void**)&resetable, num * sizeof(bool));
		gpuErrchk(cudaPeekAtLastError());
		this->num = num;
	}
	void copyFrom(const SPRING& other, cudaStream_t stream = (cudaStream_t)0) { // assuming we have enough streams
		cudaMemcpyAsync(k, other.k, num * sizeof(double), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(rest, other.rest, num * sizeof(double), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(damping, other.damping, num * sizeof(double), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(edge, other.edge, num * sizeof(Vec2i), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(resetable, other.resetable, num * sizeof(bool), cudaMemcpyDefault, stream);
		gpuErrchk(cudaPeekAtLastError());
		//this->num = other.num;
	}
};

// for displaying triangle mesh
struct TRIANGLE { 
	Vec3i* triangle = nullptr; // triangle indices
	Vec3d* vertex_normal = nullptr;// vertex normal calculated from the triangle
	int num = 0; // number of triangles
	inline int size() { return num; }
	TRIANGLE(){}
	TRIANGLE(int num, bool on_host) {
		init(num, on_host);
	}
	/* initialize and copy the state from other SPRING object. must keep the second argument*/
	TRIANGLE(TRIANGLE other, bool on_host, cudaStream_t stream = (cudaStream_t)0) {
		init(other.num, on_host);
		copyFrom(other, stream);
	}
	void init(int num, bool on_host = true) { // initialize
		cudaMallocFcnType allocateMemory = allocateMemoryFcn(on_host);// choose approipate malloc function
		allocateMemory((void**)&triangle, num * sizeof(Vec3i));
		gpuErrchk(cudaPeekAtLastError());
		this->num = num;
	}
	void copyFrom(const TRIANGLE& other, cudaStream_t stream = (cudaStream_t)0) { // assuming we have enough streams
		cudaMemcpyAsync(triangle, other.triangle, num * sizeof(Vec3i), cudaMemcpyDefault, stream);
		gpuErrchk(cudaPeekAtLastError());
		//this->num = other.num;
	}
};

struct RotAnchors { // the anchors that belongs to the rotational joints
	Vec2i* edge; // index of the (left,right) anchor of the joint
	Vec3d* dir; // direction of the joint,normalized
	double* theta;// the angular increment per joint update

	int* leftCoord; // the index of left coordintate (oxyz) start for all joints (flat view)
	int* rightCoord;// the index of right coordintate (oxyz) start for all joints (flat view)

	int num; // num of joint, num_anchor=num*2
	/*return number of joint*/
	inline int size() { return num; }

	RotAnchors() {}
	RotAnchors(std::vector<StdJoint> std_joints, bool on_host = true) { init(std_joints, on_host); }

	/* initialize and copy the state from other RotAnchors object. must keep the second argument*/
	RotAnchors(RotAnchors other, bool on_host, cudaStream_t stream = (cudaStream_t)0) {
		init(other.num, on_host);
		copyFrom(other, stream);
	}
	void init(int num, bool on_host) {
		this->num = num;
		cudaMallocFcnType allocateMemory = allocateMemoryFcn(on_host);// choose approipate malloc function
		allocateMemory((void**)&edge, num * sizeof(Vec2i));
		allocateMemory((void**)&dir, num * sizeof(Vec3d));
		allocateMemory((void**)&theta, num * sizeof(double));
		allocateMemory((void**)&leftCoord, num * sizeof(int));
		allocateMemory((void**)&rightCoord, num * sizeof(int));
		gpuErrchk(cudaPeekAtLastError());
	}

	void init(std::vector<StdJoint> std_joints, bool on_host = true) {
		init(std_joints.size(), on_host);

		if (on_host) { // copy the std_joints to this
			for (int joint_id = 0; joint_id < num; joint_id++)
			{
				edge[joint_id] = std_joints[joint_id].anchor;
				leftCoord[joint_id] = std_joints[joint_id].leftCoord;
				rightCoord[joint_id] = std_joints[joint_id].rightCoord;
			}
		}
	}

	void copyFrom(const RotAnchors& other, cudaStream_t stream = (cudaStream_t)0) {
		cudaMemcpyAsync(edge, other.edge, num * sizeof(Vec2i), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(dir, other.dir, num * sizeof(Vec3d), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(theta, other.theta, num * sizeof(double), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(leftCoord, other.leftCoord, num * sizeof(int), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(rightCoord, other.rightCoord, num * sizeof(int), cudaMemcpyDefault, stream);
		gpuErrchk(cudaPeekAtLastError());

	}
	void copyThetaFrom(const RotAnchors& other, cudaStream_t stream = (cudaStream_t)0) {
		gpuErrchk(cudaMemcpyAsync(theta, other.theta, num * sizeof(double), cudaMemcpyDefault, stream));
	}
};

struct RotPoints { // the points that belongs to the rotational joints
	int* massId; // index of the left mass and right mass
	// the directional anchor index of which the mass is rotated about, 
	int* anchorId;// e.g, k: the k-th anchor,left mass, -k: the k-th anchor, right mass
	int* dir; // direction: left=-1,right=+1
	int num; // the length of array "id"
	inline int size() { return num; }

	RotPoints() {}
	RotPoints(std::vector<StdJoint> std_joints, bool on_host) { init(std_joints, on_host); }
	/* initialize and copy the state from other RotPoints object. must keep the second argument*/
	RotPoints(RotPoints other, bool on_host, cudaStream_t stream = (cudaStream_t)0) {
		init(other.num, on_host);
		copyFrom(other, stream);
	}
	void init(int num, bool on_host = true) {
		this->num = num;
		// allocate on host or device
		cudaMallocFcnType allocateMemory = allocateMemoryFcn(on_host);// choose approipate malloc function	
		allocateMemory((void**)&massId, num * sizeof(int));
		allocateMemory((void**)&anchorId, num * sizeof(int));
		allocateMemory((void**)&dir, num * sizeof(int));
		gpuErrchk(cudaPeekAtLastError());
	}
	void init(std::vector<StdJoint> std_joints, bool on_host = true) {
		num = 0;
		for each (auto & std_joint in std_joints)
		{
			num += std_joint.left.size() + std_joint.right.size();
		}// get the total number of the points in all joints
		init(num, on_host);

		if (on_host) { // copy the std_joints to this
			size_t offset = 0;//offset the index by "offset"
			for (auto joint_id = 0; joint_id < std_joints.size(); joint_id++)
			{
				StdJoint& std_joint = std_joints[joint_id];

				for (auto i = 0; i < std_joint.left.size(); i++)
				{
					massId[offset + i] = std_joint.left[i];
					anchorId[offset + i] = joint_id;
					dir[offset + i] = -1;
				}
				offset += std_joint.left.size();//increment offset by num of left

				for (auto i = 0; i < std_joint.right.size(); i++)
				{
					massId[offset + i] = std_joint.right[i];
					anchorId[offset + i] = joint_id;
					dir[offset + i] = 1;
				}
				offset += std_joint.right.size();//increment offset by num of right
			}
		}
	}
	void copyFrom(const RotPoints& other, cudaStream_t stream = (cudaStream_t)0) {
		cudaMemcpyAsync(massId, other.massId, num * sizeof(int), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(anchorId, other.anchorId, num * sizeof(int), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(dir, other.dir, num * sizeof(int), cudaMemcpyDefault, stream);
		gpuErrchk(cudaPeekAtLastError());

	}
};

struct JOINT {
	RotPoints points; // mass points connected on the joints
	RotAnchors anchors; // anchor points of the rotation axis of the joints

	JOINT() {};
	JOINT(std::vector<StdJoint> std_joints, bool on_host = true) { init(std_joints, on_host); };

	/* initialize and copy the state from other JOINT object. must keep the second argument*/
	JOINT(JOINT other, bool on_host, cudaStream_t stream = (cudaStream_t)0) {
		points = RotPoints(other.points, on_host, stream);
		anchors = RotAnchors(other.anchors, on_host, stream);
	}

	void copyFrom(const JOINT& other, cudaStream_t stream = (cudaStream_t)0) {
		points.copyFrom(other.points, stream); // copy from the other points
		anchors.copyFrom(other.anchors, stream); // copy from the other anchor
	}

	void init(std::vector<StdJoint> std_joints, bool on_host = true) {
		anchors.init(std_joints, on_host);//initialize anchor
		points.init(std_joints, on_host);
	}
	// return num of joints
	inline int size() { return anchors.num; }
};


struct JointControl {

	double* joint_pos; // (measured) joint angle array in rad
	double* joint_vel; // (measured) joint speed array in rad/s
	double* joint_vel_desired; // (desired) joint speed array in rad/s
	double* joint_vel_error; // difference between joint_vel_cm and joint_vel
	double* joint_pos_error; // integral of the error between joint_vel_cm and joint_vel
	double* joint_vel_cmd; // (commended) joint speed array in rad/s

	double* max_joint_vel; // [rad/s] maximum joint speed
	double* max_joint_vel_error; // [rad/s] maximum joint speed error
	double* max_joint_pos_error; // [rad/s] maximum joint position error
	double k_vel = 0.5; // coefficient for speed control
	double k_pos = 0.2; // coefficient for position control

	int num = 0;
	inline int size() { return num; }
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
		cudaMallocFcnType allocateMemory = allocateMemoryFcn(on_host);// choose approipate malloc function
		allocateMemory((void**)&joint_pos, num * sizeof(double));//initialize joint angle (measured) array 
		allocateMemory((void**)&joint_vel, num * sizeof(double));//initialize joint speed (measured) array 
		allocateMemory((void**)&joint_vel_desired, num * sizeof(double));//initialize joint speed (desired) array 
		allocateMemory((void**)&joint_vel_error, num * sizeof(double));//initialize joint speed error array 
		allocateMemory((void**)&joint_pos_error, num * sizeof(double));//initialize joint speed error integral array 
		allocateMemory((void**)&joint_vel_cmd, num * sizeof(double));//initialize joint speed (commended) array 

		allocateMemory((void**)&max_joint_vel, num * sizeof(double));//initialize maximum joint speed array 
		allocateMemory((void**)&max_joint_vel_error, num * sizeof(double));//initialize maximum joint speed error array 
		allocateMemory((void**)&max_joint_pos_error, num * sizeof(double));//initialize maximum joint position error array 

		gpuErrchk(cudaPeekAtLastError());
		this->num = num;
	}

	void copyFrom(const JointControl& other, cudaStream_t stream = (cudaStream_t)0) {
		cudaMemcpyAsync(joint_pos, other.joint_pos, num * sizeof(double), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(joint_vel, other.joint_vel, num * sizeof(double), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(joint_vel_desired, other.joint_vel_desired, num * sizeof(double), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(joint_vel_error, other.joint_vel_error, num * sizeof(double), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(joint_pos_error, other.joint_pos_error, num * sizeof(double), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(joint_vel_cmd, other.joint_vel_cmd, num * sizeof(double), cudaMemcpyDefault, stream);

		cudaMemcpyAsync(max_joint_vel, other.max_joint_vel, num * sizeof(double), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(max_joint_vel_error, other.max_joint_vel_error, num * sizeof(double), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(max_joint_pos_error, other.max_joint_pos_error, num * sizeof(double), cudaMemcpyDefault, stream);

		gpuErrchk(cudaPeekAtLastError());
		// todo make max_joint_vel etc a vector
	}

	void setMaxJointSpeed(double max_joint_vel) {
		for (size_t i = 0; i < num; i++)
		{
			this->max_joint_vel[i] = max_joint_vel;
			max_joint_vel_error[i] = max_joint_vel / k_vel;
			max_joint_pos_error[i] = max_joint_vel / k_pos;
		}

	}
	void reset(MASS& mass, JOINT& joint) {
		for (int i = 0; i < num; i++) {//TODO change this...
			joint_vel_cmd[i] = 0.;
			joint_vel[i] = 0.;
			/*joint_pos[i] = 0.;*/
			joint_vel_desired[i] = 0.;
			joint_vel_error[i] = 0.;
			joint_pos_error[i] = 0.;

			Vec2i anchor_edge = joint.anchors.edge[i];
			Vec3d rotation_axis = (mass.pos[anchor_edge.y] - mass.pos[anchor_edge.x]).normalize();
			//Vec3d x_left = mass.pos[joint.anchors.leftCoord[i] + 1] - mass.pos[joint.anchors.leftCoord[i]];//oxyz
			//Vec3d x_right = mass.pos[joint.anchors.rightCoord[i] + 1] - mass.pos[joint.anchors.rightCoord[i]];//oxyz
			//joint_pos[i] = signedAngleBetween(x_left, x_right, rotation_axis); //joint angle in [-pi,pi]
			//
			Vec3d y_left = mass.pos[joint.anchors.leftCoord[i] + 2] - mass.pos[joint.anchors.leftCoord[i]];//oxyz
			Vec3d y_right = mass.pos[joint.anchors.rightCoord[i] + 2] - mass.pos[joint.anchors.rightCoord[i]];//oxyz
			joint_pos[i] = signedAngleBetween(y_left, y_right, rotation_axis); //joint angle in [-pi,pi]
		}
	}
	/*update the jointcontrol state, ndt is the delta time between jointcontrol update*/
	void update(MASS& mass, JOINT& joint, double ndt) {
		////#pragma omp parallel for simd
		for (int i = 0; i < joint.anchors.size(); i++) // compute joint angles and angular velocity
		{
			Vec2i anchor_edge = joint.anchors.edge[i];
			Vec3d rotation_axis = (mass.pos[anchor_edge.y] - mass.pos[anchor_edge.x]).normalize();
			//Vec3d x_left = mass.pos[joint.anchors.leftCoord[i] + 1] - mass.pos[joint.anchors.leftCoord[i]];//oxyz
			//Vec3d x_right = mass.pos[joint.anchors.rightCoord[i] + 1] - mass.pos[joint.anchors.rightCoord[i]];//oxyz
			//double angle = signedAngleBetween(x_left, x_right, rotation_axis); //joint angle in [-pi,pi]
			//
			Vec3d y_left = mass.pos[joint.anchors.leftCoord[i] + 2] - mass.pos[joint.anchors.leftCoord[i]];//oxyz
			Vec3d y_right = mass.pos[joint.anchors.rightCoord[i] + 2] - mass.pos[joint.anchors.rightCoord[i]];//oxyz
			double angle = signedAngleBetween(y_left, y_right, rotation_axis); //joint angle in [-pi,pi]

			// assuming delta_angle is within (-pi,pi)
			double delta_angle = angle - joint_pos[i];
			if (delta_angle > M_PI) {
				joint_vel[i] = (delta_angle - 2 * M_PI) / ndt;
			}
			else if (delta_angle > -M_PI) {
				joint_vel[i] = delta_angle / ndt;
			}
			else {// delta_angle <= M_PI
				joint_vel[i] = (delta_angle + 2 * M_PI) / ndt;
			}
			joint_pos[i] = angle;

			// update joint_vel_cmd with position PD control
			//joint_vel_cmd[i] = joint_vel_desired[i];//feed-forward control
			joint_vel_error[i] = joint_vel_desired[i] - joint_vel[i];
			joint_pos_error[i] += joint_vel_error[i]; // simple proportional control

			if (joint_pos_error[i] > max_joint_pos_error[i]) { joint_pos_error[i] = max_joint_pos_error[i]; }
			else if (joint_pos_error[i] < -max_joint_pos_error[i]) { joint_pos_error[i] = -max_joint_pos_error[i]; }
			
			joint_vel_cmd[i] = k_vel * joint_vel_error[i] + k_pos * joint_pos_error[i];
			//joint_vel_cmd[i] = k_vel * joint_vel_error[i];

			if (joint_vel_cmd[i] > max_joint_vel[i]) { joint_vel_cmd[i] = max_joint_vel[i]; }
			else if (joint_vel_cmd[i] < -max_joint_vel[i]) { joint_vel_cmd[i] = -max_joint_vel[i]; }
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

	/// <summary>
	/// initialize the rigidbody, assuming the coordinate frame is at index id_start
	/// </summary>
	/// <param name="mass"> MASS struct storing the coordinate frame of the rigidbody</param>
	/// <param name="id_start"> start index of the coordinate frame</param>
	/// <param name="w"> initial angular velocity in body frame</param>
	void init(const MASS& mass, const int& id_start,Vec3d w = Vec3d(0,0,0)) {
		pos = mass.pos[id_start];
		vel = mass.vel[id_start];
		acc = mass.acc[id_start];
		Vec3d ux = (mass.pos[id_start + 1] - pos).normalize();
		Vec3d uy = mass.pos[id_start + 2] - pos;
		uy = (uy - uy.dot(ux) * ux).normalize();
		Vec3d uz = cross(ux, uy);
		rot = Mat3d(ux, uy, uz, false);
		this->ang_vel = w;
	}
	void update(const MASS& mass, const int& id_start,const double& dt) {
		
		//vel = mass.vel[id_start]
		//acc = mass.acc[id_start];
		//pos = mass.pos[id_start];
		// instead of directly estimating the acc,
		// estimate from vel backward difference

		Vec3d pos_new = mass.pos[id_start];
		Vec3d vel_new = (pos_new - pos) / dt;
		//Vec3d vel_new = mass.vel[id_start];
		acc = (vel_new - vel) / dt; 
		vel = vel_new;
		//pos = mass.pos[id_start];
		pos = pos_new;
		
		Vec3d ux = (mass.pos[id_start + 1] - pos).normalize();
		Vec3d uy = mass.pos[id_start + 2] - pos;
		uy = (uy - uy.dot(ux) * ux).normalize();
		Vec3d uz = cross(ux, uy).normalize();
		Mat3d rot_new = Mat3d(ux, uy, uz, false);

 		ang_vel = Mat3d::angularVelocityFromRotation(rot, rot_new, dt, true);

		rot = rot_new;
	}
};



class Simulation {
public:
	double dt = 0.0001;
	double T = 0; //simulation time
	Vec3d global_acc = Vec3d(0, 0, 0); // global acceleration

	int id_restable_spring_start = 0; // resetable springs start index (inclusive)
	int id_resetable_spring_end = 0; // resetable springs start index (exclusive)

	int id_oxyz_start = 0;// coordinate oxyz start index (inclusive)
	int id_oxyz_end = 0; // coordinate oxyz end index (exclusive)



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

	// host (backup);
	MASS backup_mass;
	SPRING backup_spring;
	JOINT backup_joint;

	void backupState();//backup the robot mass/spring/joint state
	void resetState();// restore the robot mass/spring/joint state to the backedup state


	// state report
	bool RUNNING = false;
	bool STARTED = false;
	bool ENDED = false; // flag is set true when ~Simulation() is called
	bool GPU_DONE = false;

	// control
	bool RESET = false;// reset flag
	bool SHOULD_RUN = true;
	bool SHOULD_END = false;
	bool GRAPHICS_SHOULD_END = false; // a flag to notifiy the graphics thread to end
	bool GRAPHICS_ENDED = false; // a flag set by the graphics thread to notify its termination

	std::mutex mutex_running;
	std::condition_variable cv_running;

	cudaStream_t stream[NUM_CUDA_STREAM]; // cuda stream:https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution

	//Simulation();
	Simulation(size_t num_mass, size_t num_spring, size_t num_joint, size_t num_triangle);
	~Simulation();

	void getAll();
	void setAll();
	void setMass();
	// Global constraints (can be rendered)
	// creates half-space ax + by + cz < d
	void createPlane(const Vec3d& abc, const double d, const double FRICTION_K = 0, const double FRICTION_S = 0);
	void createBall(const Vec3d& center, const double r); // creates ball with radius r at position center
	void clearConstraints(); // clears global constraints only

	void setBreakpoint(const double time); // tell the program to stop at a fixed time (doesn't hang).

	void start();


	void pause(const double t);//pause the simulation at (simulation) time t [s]


	void resume();

	void update_physics();
	void update_graphics();
	void execute(); // same as above but w/out reset

#ifdef DEBUG_ENERGY
	double energy(); //compute the total energy of the system
	double energy_start;
	double energy_deviation_max = 0;
#endif // DEBUG_ENERGY


#ifdef UDP
	//Todo
public:
	std::string ip_remote = "127.0.0.1"; // remote ip
	int port_remote = 32000; // remote port
	int port_local = 32001;

	UdpServer udp_server;
#endif //UDP

private:
	void waitForEvent();
	void freeGPU();
	inline void updateCudaParameters();
	inline int computeBlocksPerGrid(const int threadsPerBlock, const int num);//helper function to compute blocksPerGrid

	std::thread thread_physics_update;
#ifdef GRAPHICS
	std::thread thread_graphics_update;
#endif //GRAPHICS
	std::set<double> bpts; // list of breakpoints


	int massBlocksPerGrid; // blocksPergrid for mass update
	int springBlocksPerGrid; // blocksPergrid for spring update
	int jointBlocksPerGrid;// blocksPergrid for joint rotation
#ifdef GRAPHICS
	int triangleBlocksPerGrid; // blocksPergrid for viewing triangles
#endif //GRAPHICS

	std::vector<Constraint*> constraints;
	thrust::device_vector<CudaContactPlane> d_planes; // used for constraints
	thrust::device_vector<CudaBall> d_balls; // used for constraints

	CUDA_GLOBAL_CONSTRAINTS d_constraints;
	bool SHOULD_UPDATE_CONSTRAINT = true; // a flag indicating whether constraint should be updated

#ifdef GRAPHICS
public:
	void setViewport(const Vec3d& camera_position, const Vec3d& target_location, const Vec3d& up_vector);
	void moveViewport(const Vec3d& displacement);

	GLFWwindow* window;
	int framebuffer_width, framebuffer_height; // the width and height of the framebuffer

	GLuint VertexArrayID; // handle for the vertex array object
	GLuint programID;  // handle for the shader program
	GLuint MatrixID; // handel for the uniform MVP

	glm::mat4 MVP; //model-view-projection matrix
	glm::mat4 View; //view matrix
	glm::mat4 Projection; //projection matrix

	// for projection matrix 
	Vec3d camera_pos;// camera position
	Vec3d camera_dir;//camera look at direction
	Vec3d camera_up;// camera up

	Vec3d camera_href_dir;// camera horizontal reference direction, initialized in setViewport()
	double camera_h_offset = 0.5;// distance b/w target and camera in plane normal to camera_up vector 
	double camera_up_offset = 0.5; // distance b/w target and camera in camera_up direction
	double camera_yaw = 0; // rotation angle of the vector from target to camera about camera_up vector

	void computeMVP(bool update_view = true); // compute MVP

	/*------------- vertex buffer object and their device pointers--------------------*/
	GLuint vbo_vertex; // handle for vertexbuffer (mass pos)
	GLfloat* dptr_vertex = nullptr;// used in updateBuffers(), device pointer,stores positions of the vertices
	struct cudaGraphicsResource* cuda_resource_vertex;

	GLuint vbo_color; // handle for colorbuffer (color)
	GLfloat* dptr_color = nullptr; // used in updateBuffers(), device pointer,stores colors of the vertices
	struct cudaGraphicsResource* cuda_resource_color;

	GLuint vbo_edge; // handle for elementbuffer (spring)
	uint2* dptr_edge = nullptr; // used in updateBuffers(), device pointer,stores indices (line plot)
	struct cudaGraphicsResource* cuda_resource_edge;

	GLuint vbo_triangle; // handle for elementbuffer (spring)
	uint3* dptr_triangle = nullptr; // used in updateBuffers(), device pointer,stores indices (line plot)
	struct cudaGraphicsResource* cuda_resource_triangle;

	bool show_triangle = true;
	bool resize_buffers = true; // update all (vbo_vertex,vbo_color,vbo_edge, vbo_triangles) if true

	inline void updateBuffers();
	inline void updateVertexBuffers();//only update vertex (positions)
	inline void generateBuffers();
	inline void resizeBuffers();
	inline void deleteBuffers();

	void createVBO(GLuint* vbo, struct cudaGraphicsResource** vbo_res, size_t size, unsigned int vbo_res_flags, GLenum buffer_type = GL_ARRAY_BUFFER);
	void deleteVBO(GLuint* vbo, struct cudaGraphicsResource* vbo_res, GLenum buffer_type);
	void resizeVBO(GLuint* vbo, size_t size, GLenum buffer_type = GL_ARRAY_BUFFER);
	/*-------------------------------------------------------------------------------*/
	inline void draw();
	void createGLFWWindow();
	static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
	static void framebuffer_size_callback(GLFWwindow* window, int width, int height);

#endif
};


#ifdef GRAPHICS
#endif


#endif //TITAN_SIM_H