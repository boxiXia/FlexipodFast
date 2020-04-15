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


//template<class T> // alias template for pinned allocator
//using ThurstHostVec = std::vector<T, thrust::system::cuda::experimental::pinned_allocator<T>>;


int main()
{
	auto start = std::chrono::steady_clock::now();

	const int num_body = 5;//number of bodies
	Model bot("..\\src\\data.msgpack"); //defined in sim.h

	int num_mass = bot.vertices.size(); // number of mass
	int num_spring = bot.edges.size(); // number of spring

	Simulation sim(num_mass, num_spring); // Simulation object
	MASS& mass = sim.mass; // reference variable for sim.mass
	SPRING& spring = sim.spring; // reference variable for sim.spring

	sim.global_acc = Vec(0, 0, -9.8); // global acceleration
	sim.dt = 4e-5; // timestep

	double m = 1e-3;// mass per vertex
	double spring_constant = 6e2; //spring constant for silicone leg
	double spring_constant_high = spring_constant*3.5;//spring constant for rigid spring
	double spring_constant_low = spring_constant*0.2;// spring constant for resetable spring
	double spring_damping = 0.4; // damping for spring

	printf("total mass:%.2f kg\n", m * num_mass);

#pragma omp parallel for
	for (int i = 0; i < num_mass; i++)
	{
		mass.pos[i]= bot.vertices[i]; // position (Vec) [m]
		mass.color[i]= bot.colors[i]; // color (Vec) [0.0-1.0]
		mass.m[i] = m; // mass [kg]
	}
#pragma omp parallel for
	for (int i = 0; i < num_spring; i++)
	{
		spring.left[i] = bot.edges[i][0]; // the left mass index of the spring
		spring.right[i] = bot.edges[i][1]; // the right mass index of the spring
		spring.k[i] = spring_constant; // spring constant
		spring.damping[i] = spring_damping; // spring constant
		spring.rest[i] = (mass.pos[spring.left[i]] - mass.pos[spring.right[i]]).norm(); // spring rest length
	}

	// bot.idVertices: body,leg0,leg1,leg2,leg3,anchor0,anchor1,anchor2,anchor3,\
					oxyz_body,oxyz_joint0_body,oxyz_joint0_leg0,oxyz_joint1_body,oxyz_joint1_leg1,\
							  oxyz_joint2_body,oxyz_joint2_leg2,oxyz_joint3_body,oxyz_joint3_leg3,the end
	// bot.idEdges: body, leg0, leg1, leg2, leg3, anchors, rotsprings, fricsprings, oxyz_self_springs, oxyz_anchor_springs, the end

	// set higher spring constant for the robot body
	for (int i = 0; i < bot.idEdges[1]; i++)
	{
		spring.k[i] = spring_constant_high;
	}
	// set higher spring constant for the rotational joints
	for (int i = bot.idEdges[num_body]; i < bot.idEdges[num_body +1]; i++)
	{
		spring.k[i] = spring_constant_high; // joints anchors
	}
	for (int i = bot.idEdges[num_body +1]; i < bot.idEdges[num_body +2]; i++)
	{
		spring.k[i] = spring_constant_high; // joints rotation spring
	}

	sim.id_restable_spring_start = bot.idEdges[num_body + 2]; // resetable spring (frictional spring)
	sim.id_resetable_spring_end = bot.idEdges[num_body + 3];
	for (int i = sim.id_restable_spring_start; i < sim.id_resetable_spring_end; i++)
	{
		spring.k[i] = spring_constant/5.;// resetable spring, reset the rest length per dynamic update
		spring.damping[i] = spring_damping*2.;
	}


	

	//// the start (inclusive) and end (exclusive) index of the anchor points
	//double id_joint_anchor_start = bot.idVertices[num_body];
	//double id_joint_anchor_end = bot.idVertices[num_body + sim.num_joint];

	//oxyz_body,oxyz_joint0_body,oxyz_joint0_leg0,oxyz_joint1_body,oxyz_joint1_leg1,\
				oxyz_joint2_body,oxyz_joint2_leg2,oxyz_joint3_body,oxyz_joint3_leg3,
	sim.id_oxyz_start = bot.idVertices[num_body + sim.num_joint];
	sim.id_oxyz_end = bot.idVertices[num_body + sim.num_joint + 1 + 2* sim.num_joint];

	double scale_down = 0.1;
	// set lower mass for the anchored coordinate systems
	for (int i = sim.id_oxyz_start; i < sim.id_oxyz_end; i++)
	{
		mass.m[i] = m * scale_down; // mass [kg]
	}

	for (int i = bot.idEdges[num_body + 3]; i < bot.idEdges[num_body + 4]; i++)
	{
		spring.k[i] = spring_constant * 2*scale_down;// oxyz_self_springs
		spring.damping[i] = spring_damping * scale_down;
	}
	for (int i = bot.idEdges[num_body + 4]; i < bot.idEdges[num_body + 5]; i++)
	{
		spring.k[i] = spring_constant * scale_down;// oxyz_anchor_springs
		spring.damping[i] = spring_damping * scale_down;
	}


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


	sim.all_joints.init(bot.Joints, true);
	sim.d_all_joints.init(bot.Joints, false);
	sim.d_all_joints.copyFrom(sim.all_joints);


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

	auto end = std::chrono::steady_clock::now();
	printf("main():Elapsed time:%d ms \n",
		std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

	return 0;
}

