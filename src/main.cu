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

#include<cuda_runtime.h>
#include<cuda_device_runtime_api.h>
#include<device_launch_parameters.h>

#include <thrust/device_vector.h>
#include<thrust/system/cuda/experimental/pinned_allocator.h>

#include <stdio.h>
#include <sstream>
#include <fstream>
#include<iostream>
#include<stdlib.h>
#include<string.h>
#include <chrono> // for time measurement

#include "shader.h"
#include "object.h"
#include "sim.h"


#define _USE_MATH_DEFINES
#include <math.h>

#include <omp.h>

#include <msgpack.hpp>

#include <thread>

#include "vec.h"
#include <complex>




template<class T> // alias template for pinned allocator
using ThurstHostVec = std::vector<T, thrust::system::cuda::experimental::pinned_allocator<T>>;


//struct D_Joint {
//	left
// };

struct StdJoint {
	std::vector<int> left;// the indices of the left points
	std::vector<int> right;// the indices of the right points
	std::vector<int> anchor;// the 2 indices of the anchor points: left_anchor_id,right_anchor_id
	MSGPACK_DEFINE(left, right, anchor);
};
class Model {
public:
	std::vector<std::vector<double> > vertices;// the mass xyzs
	std::vector<std::vector<int> > edges;//the spring ids
	std::vector<int> idVertices;// the edge id of the vertices
	std::vector<int> idEdges;// the edge id of the springs
	std::vector<std::vector<double> > colors;// the mass xyzs
	std::vector<StdJoint> Joints;// the joints
	MSGPACK_DEFINE(vertices, edges, idVertices, idEdges, colors, Joints); // write the member variables that you want to pack

	Model() {
	}

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



//__global__ void dummykernel(MASS d_mass,int num_mass) {
//	int i = blockDim.x * blockIdx.x + threadIdx.x; // todo: change to grid-strided loop
//	if (i < num_mass) {
//		d_mass.vel[i][0] = (double)i; // todo check if this is needed
//		printf("%d", i);
//	}
//}


int main()
{
	auto start = std::chrono::steady_clock::now();

	Model bot("..\\src\\data.msgpack");

	int num_mass = bot.vertices.size();
	int num_spring = bot.edges.size();


	Simulation sim(num_mass, num_spring);
	MASS& mass = sim.mass;
	SPRING& spring = sim.spring;

	sim.global_acc = Vec(0, 0, -9.8);
	sim.dt = 4e-5;

	double m = 5e-2;// mass per vertex
	double spring_constant = 3e4;
	double spring_damping = 20.;

#pragma omp parallel for
	for (int i = 0; i < num_mass; i++)
	{
		mass.pos[i]= bot.vertices[i];
		mass.color[i]= bot.colors[i];
		mass.m[i] = m; // mass [kg]
	}
#pragma omp parallel for
	for (int i = 0; i < num_spring; i++)
	{
		spring.left[i] = bot.edges[i][0];
		spring.right[i] = bot.edges[i][1];
		spring.k[i] = spring_constant; // spring constant
		spring.damping[i] = spring_damping;
		spring.rest[i] = (mass.pos[spring.left[i]] - mass.pos[spring.right[i]]).norm();
	}


	// set higher spring constant for the robot body
	for (int i = 0; i < bot.idEdges[1]; i++)
	{
		spring.k[i] = 1e5;
	}
	// set higher spring constant for the rotational joints
	for (int i = bot.idEdges[5]; i < bot.idEdges[5+sim.num_joint]; i++)
	{
		spring.k[i] = 1e5;
	}
	for (int i = bot.idEdges[5 + sim.num_joint]; i < bot.edges.size(); i++)
	{
		spring.k[i] = 1e4;// resetable spring
		spring.damping[i] = spring_damping*2.;
	}
	sim.id_restable_spring_start = bot.idEdges[5 + sim.num_joint]; // resetable spring
	sim.id_resetable_spring_end = bot.edges.size();

#pragma omp parallel for
	for (int i = 0; i < sim.num_joint; i++)
	{
		auto& std_joint = bot.Joints[i];
		auto& joint = sim.joints[i];
		auto& d_joint = sim.d_joints[i];
		joint.init(std_joint.left.size(), std_joint.right.size(), true);//init the host
		d_joint.init(std_joint.left.size(), std_joint.right.size(), false);//init the device, TODO
		for (int k = 0; k < std_joint.left.size(); k++)
		{ // note that #left may not equal # right
			joint.left[k] = std_joint.left[k];
		}
		for (int k = 0; k < std_joint.right.size(); k++)
		{
			joint.right[k] = std_joint.right[k];
		}
		joint.anchor[0] = std_joint.anchor[0];
		joint.anchor[1] = std_joint.anchor[1];
	}

	// set max speed for each joint
	sim.max_joint_speed = 200. / 60. * 2 * 3.1415926 * sim.dt;//200 rpm

	sim.setViewport(Vec(0, -0.3, 0.3), Vec(0, 0, 0), Vec(0, 0, 1));
	//sim.setViewport(Vec(.4, -0., .4), Vec(0, -0., -0), Vec(0, 0, 1));

	// our plane has a unit normal in the z-direction, with 0 offset.
	//sim.createPlane(Vec(0, 0, 1), 0, 0.5, 0.55);
	sim.createPlane(Vec(0, 0, 1), 0, 0.6, 0.65);


	//double runtime = 120;
	//sim.setBreakpoint(runtime);
	

	sim.start();

	//auto start = std::chrono::steady_clock::now();

	while (sim.RUNNING) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}

	//auto end = std::chrono::steady_clock::now();
	//std::cout << "Elapsed time:"
	//	<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
	//	<< " ms" << std::endl;


	auto end = std::chrono::steady_clock::now();
	printf("main():Elapsed time:%d ms \n",
		std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());



	//std::complex<double> a;

	return 0;
}

