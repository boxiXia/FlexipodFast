#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#endif // !CUDA_API_PER_THREAD_DEFAULT_STREAM

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

#include<algorithm>

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
	// for time measurement
	auto start = std::chrono::steady_clock::now();

	std::cout << "current working dir: " << getWorkingDir() << "\n";
	std::cout << "program dir: " << getProgramDir() << "\n";


	Model bot(getProgramDir()+"\\flexipod_12dof.msgpack"); //defined in sim.h
	//Model bot(getProgramDir() + "\\flexipod_4dof.msgpack"); //defined in sim.h


	const size_t num_body = bot.idVertices.size() - 3;//number of bodies
	const size_t num_mass = bot.vertices.size(); // number of mass
	const size_t num_spring = bot.edges.size(); // number of spring
	const size_t num_triangle = bot.triangles.size();// number of triangle
	const size_t num_joint = bot.joints.size();//number of rotational joint

	Simulation sim(num_mass, num_spring,num_joint,num_triangle); // Simulation object

	MASS& mass = sim.mass; // reference variable for sim.mass
	SPRING& spring = sim.spring; // reference variable for sim.spring
	TRIANGLE triangle = sim.triangle;// reference variable for sim.triangle

	sim.dt = 4e-5; // timestep
	//sim.dt = 5e-5; // timestep

	//constexpr double radius_poisson = 12.5 * 1e-3;
	const double  radius_poisson = bot.radius_poisson;
	

	const double radius_knn = radius_poisson * sqrt(3.0);
	const double min_radius = radius_poisson * 0.5;

	const double m = 0.10* radius_poisson;// mass per vertex
	//const double m = 2.5/(double)num_mass;// mass per vertex

	const double spring_constant = m*6e6; //spring constant for silicone leg
	const double spring_damping = m*5e2; // damping for spring

	//const double spring_constant = m * 22e6; //spring constant for silicone leg
	//const double spring_damping = m * 3e2; // damping for spring
	//constexpr double scale_rigid = 1;// scaling factor rigid


	constexpr double scale_rigid = 2.5;// scaling factor rigid
	constexpr double scale_soft = 2; // scaling factor soft

	constexpr double scale_joint_k = 4.0; // scaling factor for the joint spring constant
	constexpr double scale_joint_m = 2.0; // scaling factor for the joint mass
	constexpr double scale_joint_damping = 4.0; // scaling factor for the joint spring damping

	//const double scale_low = 0.5; // scaling factor low
	constexpr double scale_probe = 0.08; // scaling factor for the probing points, e.g. coordinates

	const double spring_constant_rigid = spring_constant * scale_rigid;//spring constant for rigid spring
	const double spring_constant_soft = spring_constant * scale_soft;//spring constant for soft spring

	const double spring_constant_restable = spring_constant * scale_joint_k; // spring constant for resetable spring
	const double spring_damping_restable = spring_damping * scale_joint_damping; // spring damping for resetable spring

	//const double spring_constant_restable = 0; // spring constant for resetable spring
	//const double spring_damping_restable = 0; // spring damping for resetable spring

	// spring coefficient for the probing springs, e.g. coordinates
	const double m_probe = m * scale_probe;//mass for the probe
	const double spring_constant_probe_anchor = spring_constant * scale_probe* scale_rigid; // spring constant for coordiates anchor springs
	const double spring_constant_probe_self = spring_constant * scale_probe* scale_rigid; // spring constant for coordiates self springs
	const double spring_damping_probe = spring_damping * scale_probe* scale_rigid;

//ref: https://bisqwit.iki.fi/story/howto/openmp/
//#pragma omp parallel for
#pragma omp simd
	for (int i = 0; i < num_triangle; i++)
	{
		triangle.triangle[i] = bot.triangles[i];
		//if (i == 0) {
		//	std::cout << omp_get_num_threads() << std::endl;
		//}
	}

#pragma omp simd
	for (int i = 0; i < num_mass; i++)
	{
		mass.pos[i] = bot.vertices[i]; // position (Vec3d) [m]
		mass.color[i] = bot.colors[i]; // color (Vec3d) [0.0-1.0]
		mass.m[i] = m; // mass [kg]
		mass.constrain[i] = bot.isSurface[i];// set constraint to true for suface points, and false otherwise
	}
#pragma omp simd
	for (int i = 0; i < num_spring; i++)
	{
		spring.edge[i] = bot.edges[i]; // the (left,right) mass index of the spring
		spring.damping[i] = spring_damping; // spring constant
		spring.rest[i] = (mass.pos[spring.edge[i].x] - mass.pos[spring.edge[i].y]).norm(); // spring rest length
		// longer spring will have a smalller influence, https://ccrma.stanford.edu/~jos/pasp/Young_s_Modulus_Spring_Constant.html
		spring.k[i] = spring_constant_soft * radius_knn / std::max(spring.rest[i], min_radius); // spring constant
		//spring.k[i] = spring_constant; // spring constant
		spring.resetable[i] = false; // set all spring as non-resetable
	}


	/*bot.idVertices: body,leg0,leg1,leg2,leg3,anchor,coord,the end
	 bot.idEdges: body, leg0, leg1, leg2, leg3, anchors, rotsprings, fricsprings, oxyz_self_springs, oxyz_anchor_springs, the end */


	 //// set higher mass value for robot body
	 //for (int i = bot.idVertices[0]; i < bot.idVertices[1]; i++)
	 //{
	 //	mass.m[i] = m*1.8; // accounting for addional mass for electornics
	 //}
	 //// set lower mass value for leg
	 //for (int i = bot.idVertices[1]; i < bot.idVertices[num_body]; i++)
	 //{
	 //	mass.m[i] = m * 0.3; // 80% infill,no skin
	 //}

	 // set the mass value for joint
#pragma omp simd
	for (int i = 0; i < bot.joints.size(); i++)
	{
		for each (int j in bot.joints[i].left)
		{
			mass.m[j] = m * scale_joint_m;
		}
		for each (int j in bot.joints[i].right)
		{
			mass.m[j] = m * scale_joint_m;
		}
	}


	//// set higher spring constant for the robot body
	//for (int i = 0; i < bot.idEdges[1]; i++)
	//{
	//	spring.k[i] = spring.k[i] * scale_rigid;
	//}

	// set higher spring constant for the rotational joints
	for (int i = bot.idEdges[num_body]; i < bot.idEdges[num_body + 1]; i++)
	{
		spring.k[i] = spring_constant_rigid; // joints anchors
	}
	for (int i = bot.idEdges[num_body + 1]; i < bot.idEdges[num_body + 2]; i++)
	{
		spring.k[i] = spring_constant_rigid; // joints rotation spring
		//spring.k[i] = spring_constant; // joints rotation spring
		//spring.damping[i] = spring_damping_restable;
	}

	sim.id_restable_spring_start = bot.idEdges[num_body + 2]; // resetable spring (frictional spring)
	sim.id_resetable_spring_end = bot.idEdges[num_body + 3];
	for (int i = sim.id_restable_spring_start; i < sim.id_resetable_spring_end; i++)
	{
		spring.k[i] = spring_constant_restable;// resetable spring, reset the rest length per dynamic update
		spring.damping[i] = spring_damping_restable;
		spring.resetable[i] = true;
	}

	sim.id_part_end = bot.idVertices[num_body];// index where the parts m ends
	sim.id_oxyz_start = bot.idVertices[num_body + 1];
	sim.id_oxyz_end = bot.idVertices[num_body + 2];


	// set lower mass for the anchored coordinate systems
	for (int i = sim.id_oxyz_start; i < sim.id_oxyz_end; i++)
	{
		mass.m[i] = m_probe; // mass [kg]
	}

	for (int i = bot.idEdges[num_body + 3]; i < bot.idEdges[num_body + 4]; i++)
	{
		spring.k[i] = spring_constant_probe_self;// oxyz_self_springs
		spring.damping[i] = spring_damping_probe;
	}
	for (int i = bot.idEdges[num_body + 4]; i < bot.idEdges[num_body + 5]; i++)
	{
		spring.k[i] = spring_constant_probe_anchor;// oxyz_anchor_springs
		spring.damping[i] = spring_damping_probe;
	}

	sim.joint.init(bot.joints, true);
	sim.d_joint.init(bot.joints, false);
	sim.d_joint.copyFrom(sim.joint);


	// print out mass statistics
	double total_mass = 0;
//#pragma omp parallel for shared (total_mass,mass.m) reduction(+:total_mass)
	for (int i = 0; i < num_mass; i++) { total_mass += mass.m[i]; }
	printf("total mass:%.2f [kg]\n",total_mass);

	std::vector<double> part_mass(num_body,0); // vector of size num_body, with values 0
#pragma omp parallel for
	for (int k = 0; k < num_body; k++)
		for (int i = bot.idVertices[k]; i < bot.idVertices[k + 1]; i++)
			part_mass[k] += mass.m[i];
	printf("part mass:");
	for (const auto& i : part_mass) 
		printf("%.2f ", i);
	printf("[kg]\n");

	std::vector<double> joint_mass(num_joint, 0); // vector of size num_body, with values 0
#pragma omp parallel for
	for (int k = 0; k < num_joint; k++)
		for (const auto & i:bot.joints[k].left)
			joint_mass[k] += mass.m[i];
	printf("joint mass:");
	for (const auto& i : joint_mass)
		printf("%.2f ", i);
	printf("[kg]\n");


	// set max speed for each joint
	double max_rpm = 400;//maximun revolution per minute
	sim.joint_control.max_vel = max_rpm / 60. * 2 * M_PI;//max joint speed in rad/s
	double settling_time = 0.2;// reaches max_rpm within this time
	sim.joint_control.max_acc = sim.joint_control.max_vel / settling_time;

#ifdef GRAPHICS
	//sim.setViewport(Vec3d(-0.3, 0, 0.3), Vec3d(0, 0, 0), Vec3d(0, 0, 1));
	//sim.setViewport(Vec3d(0.6, 0, 0.3), Vec3d(0, 0, 0.2), Vec3d(0, 0, 1));
	sim.setViewport(Vec3d(1.75, -2.5, 1.0), Vec3d(1.75, 0, 0.1), Vec3d(0, 0, 1));
#endif // GRAPHICS

	// our plane has a unit normal in the z-direction, with 0 offset.
	//sim.createPlane(Vec3d(0, 0, 1), -1, 0, 0);

	sim.global_acc = Vec3d(0, 0, -9.8); // global acceleration
	sim.createPlane(Vec3d(0, 0, 1), 0, 0.9, 0.8);
	//sim.createPlane(Vec3d(0, 0, 1), 0, 0.4, 0.35);


	//double runtime = 172800;//48 hours
	//double runtime = 15;//48 hours
	//sim.setBreakpoint(runtime,true);

	sim.start();
	
	//while (sim.RUNNING) {
	//	std::this_thread::sleep_for(std::chrono::milliseconds(1));
	//}
	//sim.resume();

	auto end = std::chrono::steady_clock::now();
	printf("main():Elapsed time:%d ms \n",
		(int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

	return 0;
}

