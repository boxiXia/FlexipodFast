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

#ifdef UDP
#include "network.h"
#include <msgpack.hpp>
#endif //DUP

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

	/*allocate memory and set initial values*/
	void init(int num, bool on_host = true);

	/*copy data asynchronously from other MASS on specified cuda stream*/
	void copyFrom(const MASS& other, cudaStream_t stream = (cudaStream_t)0);

	MASS() { }
	MASS(int num, bool on_host) { init(num, on_host); }
	/* copy construct from other MASS object on specified cuda stream*/
	MASS(MASS other, bool on_host, cudaStream_t stream = (cudaStream_t)0);

	/* copy pos, vel, acc from other MASS on specified cuda stream*/
	void CopyPosVelAccFrom(MASS& other, cudaStream_t stream = (cudaStream_t)0);

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

struct SPRING {
	double* compliance = nullptr; // spring constant (N/m)
	double* rest = nullptr; // spring rest length (meters)
	double* damping = nullptr; // damping on the masses.
	Vec2i* edge = nullptr;// (left,right) mass indices of the spring
	bool* resetable = nullptr; // a flag indicating whether to reset every dynamic update
	int* joint_id = nullptr; // pointer to joint spring id
	int num = 0;
	inline int size() { return num; }

	/* initialilze and allocate data*/
	void init(int num, bool on_host = true);

	/*copy data asynchronously from other SPRING on specified cuda stream*/
	void copyFrom(const SPRING& other, cudaStream_t stream = (cudaStream_t)0);

	SPRING() {}
	SPRING(int num, bool on_host) { init(num, on_host); }
	/* copy construct from other SPRING object on specified cuda stream*/
	SPRING(SPRING other, bool on_host, cudaStream_t stream = (cudaStream_t)0);
};

// for displaying triangle mesh
struct TRIANGLE {
	Vec3i* triangle = nullptr; // triangle indices
	int num = 0; // number of triangles
	inline int size() { return num; }
	TRIANGLE() {}
	TRIANGLE(int num, bool on_host) { init(num, on_host); }
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

// enum for the JointControl mode
enum class JointControlMode {
	vel = 0, // velocity control mode
	pos = 1, // position control mode
};

struct JOINT;

struct CUDA_JOINT {
	/*--------------- joint-----------------------*/
	int num; // num of joint
	Vec2i* anchor; // index of the (left,right) anchor of the joint
	double* pos; // (measured) joint position [rad] (-pi,pi]
	double* delta_pos;// the joint position increment per joint update
	double* torque; // computed torque
	/*--------------- edges -----------------------*/
	int edge_num; // number of edges
	int* edge_id; // spring index of the left mass and right mass
	int* edge_joint_id; // e.g, k: index of the joint the edge belongs to
	Vec3d* edge_c; // edge constant (c0,c1,phase)

	/*initialize & allocate memory. num:#of joint, edge_num:#of joint edges*/
	void init(int num, int edge_num);

	/*copy data asynchronously from other JOINT on specified cuda stream*/
	void copyFrom(const JOINT& other, cudaStream_t stream = (cudaStream_t)0);
};

struct JOINT {
	/*--------------- joint-----------------------*/
	int num; // num of joint
	Vec2i* anchor; // index of the (left,right) anchor of the joint
	double* delta_pos;// the joint position increment per joint update
	double* pos; // (measured) joint position [rad] (-pi,pi]
	double* vel; // (measured) joint speed array in rad/s
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
	/* copy construct from other JOINT object on specified cuda stream*/
	JOINT(JOINT other, bool on_host, cudaStream_t stream = (cudaStream_t)0);

	/*initialize & allocate memory. num:#of joint, vert_num:#of joint vertecies, edge_num:#of joint edges*/
	void init(int num, int vert_num, int edge_num, bool on_host);

	/*initialze from given model*/
	void init(const Model& robot, bool on_host = true);

	/*copy data asynchronously from other JOINT on specified cuda stream*/
	void copyFrom(const JOINT& other, cudaStream_t stream = (cudaStream_t)0);

	/*copy data asynchronously from other CUDA_JOINT on specified cuda stream*/
	void copyFrom(const CUDA_JOINT& other, cudaStream_t stream = (cudaStream_t)0);

	/*return the k-th joint noramlized rotation axis given mass_pos, where jid = k */
	Vec3d computeRotationAxis(const Vec3d* mass_pos, const int jid) {
		Vec3d rotation_axis = (mass_pos[anchor[jid].y] - mass_pos[anchor[jid].x]).normalize();
		return rotation_axis;
	}

	/* compute the k-th joint position given mass_pos, where jid = k*/
	double computeJointPos(const Vec3d* mass_pos, const int jid, const Vec3d rotation_axis) {
		// 0  1  2  3  4  5
		// x  y  z -x -y -z
		Vec3d y_left = mass_pos[left_coord[jid] + 1] - mass_pos[left_coord[jid] + 4];//(x,y,z,-x,-y,-z)
		Vec3d y_right = mass_pos[right_coord[jid] + 1] - mass_pos[right_coord[jid] + 4];//(x,y,z,-x,-y,-z)
		double pos = signedAngleBetween(y_left, y_right, rotation_axis); //joint angle in [-pi,pi]
		return pos; // joint pos
	}

	void copyPosFrom(const JOINT& other, cudaStream_t stream = (cudaStream_t)0) {
		gpuErrchk(cudaMemcpyAsync(pos, other.pos, num * sizeof(double), cudaMemcpyDefault, stream));
	}


	// joint control
	JointControlMode mode = JointControlMode::vel;
	double max_vel; // [rad/s] maximum joint speed
	double max_acc; // [rad/s^2] maximum joint acceleration

	/*update the jointcontrol state, n is the number of queued kernels in the dynamics thread,
	dt is the delta time per queued kernels, i.e.: n*dt is the delta time between jointcontrol update*/
	void update(const MASS& mass, int n, double dt) {
		//#pragma omp simd
		for (int i = 0; i < num; i++) // compute joint angles and angular velocity
		{
			Vec3d rotation_axis = computeRotationAxis(mass.pos, i);
			double pos_now = computeJointPos(mass.pos, i, rotation_axis);

			double d_pos = pos_now - pos[i]; // joint position difference
			// assuming delta_angle is within (-pi,pi)
			// there may be a huge chage from -pi to pi, but really it only moves a little bit
			clampPeroidicInplace(d_pos, -M_PI, M_PI);
			double ndt = n * dt; // delta time between jointcontrol update
			vel[i] = d_pos / ndt;
			pos[i] = pos_now;

			// update cmd with position PD control: clamp to (-max_vel,max_vel)
			double dv = max_acc * ndt;;
			double cmd; // velocity command proxy
			switch (mode)//switch controller mode
			{
			case JointControlMode::vel:
				clampInplace(vel_desired[i], -max_vel, max_vel);
				cmd = vel_desired[i];
				clampInplace(cmd, vel[i] - dv, vel[i] + dv);
				//vel_desired[i] = 0.99 * vel_desired[i]; // can optionally add damping to vel_desired
				break;
			case JointControlMode::pos:
				clampPeroidicInplace(pos_desired[i], -M_PI, M_PI);
				cmd = pos_desired[i] - pos[i]; // position error
				clampPeroidicInplace(cmd, -M_PI, M_PI);
				cmd = 10 * cmd - 0.1 * vel[i]; // velocity command proxy
				clampInplace(cmd, vel[i] - dv, vel[i] + dv);
				clampInplace(cmd, -max_vel, max_vel);
				break;
			}
			delta_pos[i] = cmd * dt; // set joint position update per kernel launch
		}
	}

	void updateControlMode(const JointControlMode& control_mode) {
		if (mode != control_mode) {
			mode = control_mode; // update mode to control_mode
			switch (mode) {
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

	void resetControl(const MASS& mass) {
		size_t nbytes = num * sizeof(double);
		memset(vel, 0, nbytes);//vel[i] = 0.;
		memset(vel_desired, 0, nbytes);//vel_desired[i] = 0.;
		for (int i = 0; i < num; i++) {//TODO change this...
			Vec3d rotation_axis = computeRotationAxis(mass.pos, i);
			pos[i] = computeJointPos(mass.pos, i, rotation_axis);
			pos_desired[i] = pos[i];//keep it still
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
	/*store text info to out array with known size.
	  return The number of characters that would have been written*/
	int print(char out[], int size) {
		int n = snprintf(out, size,
			"com pos %+6.2f %+6.2f %+6.2f |%6.2f|\n"
			"com vel %+6.2f %+6.2f %+6.2f |%6.2f|\n"
			"com acc %+6.2f %+6.2f %+6.2f |%6.2f|\n"
			"ang vel %+6.2f %+6.2f %+6.2f |%6.2f|\n"
			"rotation:\n"
			"%+7.3f %+7.3f %+7.3f\n"
			"%+7.3f %+7.3f %+7.3f\n"
			"%+7.3f %+7.3f %+7.3f\n",
			pos.x, pos.y, pos.z, pos.norm(),
			vel.x, vel.y, vel.z, vel.norm(),
			acc.x, acc.y, acc.z, acc.norm(),
			ang_vel.x, ang_vel.y, ang_vel.z, ang_vel.norm(),
			rot.m00, rot.m01, rot.m02,
			rot.m10, rot.m11, rot.m12,
			rot.m20, rot.m21, rot.m22
		);
		return n;
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
	RigidBody body; // mainbody
	// device
	MASS d_mass;
	SPRING d_spring;
	TRIANGLE d_triangle; // a flat view of all triangles
	CUDA_JOINT d_joint;

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
	// use this->udp_server.setaddress to set ip_local,port_local,ip_remote,port_remote
	asioUdpServer udp_server = asioUdpServer("127.0.0.1", 33301, "127.0.0.1", 33300);//defualt

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
	glm::vec3 camera_target_offset{ 0,0,0 };


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