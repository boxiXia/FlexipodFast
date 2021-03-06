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
	

	// m_total->3113g
	// for time measurement
	auto start = std::chrono::steady_clock::now();

	constexpr int num_body = 5;//number of bodies
 	Model bot("..\\src\\data.msgpack"); //defined in sim.h

	const size_t num_mass = bot.vertices.size(); // number of mass
	const size_t num_spring = bot.edges.size(); // number of spring

	const size_t num_joint = bot.Joints.size();//number of rotational joint

	Simulation sim(num_mass, num_spring); // Simulation object
	MASS& mass = sim.mass; // reference variable for sim.mass
	SPRING& spring = sim.spring; // reference variable for sim.spring
	
	
	//sim.dt = 4e-5; // timestep
	sim.dt = 5e-5; // timestep

	constexpr  double radius_poisson = 10*1e-3;

	const double radius_knn = radius_poisson * sqrt(3.0);
	constexpr double mimimun_radius = radius_poisson * 0.5;

	const double m = 6e-4;// mass per vertex
	//const double m = 2.5/(double)num_mass;// mass per vertex

	const double spring_constant =m*2.4e6; //spring constant for silicone leg
	//const double spring_damping = m*1.8e2; // damping for spring
	const double spring_damping = m * 1.5e2; // damping for spring

	//const double spring_constant = m * 1.5e6; //spring constant for silicone leg
	//const double spring_damping = m * 1.5e2; // damping for spring


	constexpr double scale_high = 2;// scaling factor high
	//const double scale_low = 0.5; // scaling factor low
	constexpr double scale_probe = 0.08; // scaling factor for the probing points, e.g. coordinates

	const double spring_constant_rigid = spring_constant* scale_high;//spring constant for rigid spring

	const double spring_constant_restable = spring_constant * scale_high; // spring constant for resetable spring
	const double spring_damping_restable = spring_damping*2.4; // spring damping for resetable spring

	//const double spring_constant_restable = 0; // spring constant for resetable spring
	//const double spring_damping_restable = 0; // spring damping for resetable spring

	// spring coefficient for the probing springs, e.g. coordinates
	const double spring_constant_probe_anchor = spring_constant * scale_probe; // spring constant for coordiates anchor springs
	const double spring_constant_probe_self = spring_constant * scale_probe* scale_high; // spring constant for coordiates self springs
	const double spring_damping_probe = spring_damping * scale_probe * scale_high;

#pragma omp parallel for
	for (int i = 0; i < num_mass; i++)
	{
		mass.pos[i]= bot.vertices[i]; // position (Vec3d) [m]
		mass.color[i]= bot.colors[i]; // color (Vec3d) [0.0-1.0]
		mass.m[i] = m; // mass [kg]
		mass.constrain[i] = bot.isSurface[i];// set constraint to true for suface points, and false otherwise
	}
#pragma omp parallel for
	for (int i = 0; i < num_spring; i++)
	{
		spring.edge[i] = bot.edges[i]; // the (left,right) mass index of the spring
		spring.damping[i] = spring_damping; // spring constant
		spring.rest[i] = (mass.pos[spring.edge[i].x] - mass.pos[spring.edge[i].y]).norm(); // spring rest length

		// longer spring will have a smalller influence
		spring.k[i] = spring_constant * radius_knn / std::max(spring.rest[i], mimimun_radius); // spring constant
		//spring.k[i] = spring_constant; // spring constant


		spring.resetable[i] = false; // set all spring as non-resetable
	}

/*bot.idVertices: body,leg0,leg1,leg2,leg3,anchor0,anchor1,anchor2,anchor3,
				oxyz_body,oxyz_joint0_body,oxyz_joint0_leg0,oxyz_joint1_body,oxyz_joint1_leg1,
				oxyz_joint2_body,oxyz_joint2_leg2,oxyz_joint3_body,oxyz_joint3_leg3,the end
 bot.idEdges: body, leg0, leg1, leg2, leg3, anchors, rotsprings, fricsprings, oxyz_self_springs, oxyz_anchor_springs, the end */
	
	// set higher mass value for robot body
	for (int i = bot.idVertices[0]; i < bot.idVertices[1]; i++)
	{
		mass.m[i] = m*1.8; // accounting for addional mass for electornics
	}
	// set lower mass value for leg
	for (int i = bot.idVertices[1]; i < bot.idVertices[1+4]; i++)
	{
		mass.m[i] = m * 0.3; // 80% infill,no skin
	}

	// set the mass value for joint
//#pragma omp parallel for
	
	for (int i = 0; i < bot.Joints.size(); i++)
	{
		for each (int j in bot.Joints[i].left)
		{
			mass.m[j] = m*1.4;
		}	
		for each (int j in bot.Joints[i].right)
		{
			mass.m[j] = m*1.4;
		}
	}
	

	// set higher spring constant for the robot body
	for (int i = 0; i < bot.idEdges[1]; i++)
	{
		//spring.k[i] = spring_constant_rigid;
		spring.k[i] *= scale_high;
	}
	// set higher spring constant for the rotational joints
	for (int i = bot.idEdges[num_body]; i < bot.idEdges[num_body +1]; i++)
	{
		spring.k[i] = spring_constant_rigid; // joints anchors
		//spring.k[i] *= scale_high;
	}
	for (int i = bot.idEdges[num_body +1]; i < bot.idEdges[num_body +2]; i++)
	{
		spring.k[i] = spring_constant_rigid; // joints rotation spring
		//spring.k[i] *= scale_high;
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

	//// the start (inclusive) and end (exclusive) index of the anchor points
	//double id_joint_anchor_start = bot.idVertices[num_body];
	//double id_joint_anchor_end = bot.idVertices[num_body + num_joint];

	/*oxyz_body,oxyz_joint0_body,oxyz_joint0_leg0,oxyz_joint1_body,oxyz_joint1_leg1,
				oxyz_joint2_body,oxyz_joint2_leg2,oxyz_joint3_body,oxyz_joint3_leg3,*/
	sim.id_oxyz_start = bot.idVertices[num_body + num_joint];
	sim.id_oxyz_end = bot.idVertices[num_body + num_joint + 1 + 2* num_joint];
	
	// set lower mass for the anchored coordinate systems
	for (int i = sim.id_oxyz_start; i < sim.id_oxyz_end; i++)
	{
		mass.m[i] = m * scale_probe; // mass [kg]
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

	sim.joint.init(bot.Joints, true);
	sim.d_joint.init(bot.Joints, false);
	sim.d_joint.copyFrom(sim.joint);


	double total_mass = 0;
#pragma omp parallel for shared (total_mass,mass.m) reduction(+:total_mass)
	for (int i = 0; i < num_mass; i++){total_mass += mass.m[i];}

	double body_mass = 0;
	for (int i = bot.idVertices[0]; i < bot.idVertices[1]; i++)
	{body_mass+= mass.m[i];}// calculate body mass
	

	double leg_mass = 0;
	for (int i = bot.idVertices[1]; i < bot.idVertices[2]; i++)
	{leg_mass += mass.m[i];}// calculate leg mass

	double joint_mass = 0;
	for each (int j in bot.Joints[0].right)
	{joint_mass+=mass.m[j];}// calculate joint mass



	printf("total mass:%.2f kg, body mass:%.2f kg, per leg mass:%.2f kg (soft part:%.2f kg)\n", 
		total_mass, body_mass, leg_mass, leg_mass - joint_mass);

	// set max speed for each joint
	double max_rpm = 600;//maximun revolution per minute
	sim.setMaxJointSpeed(max_rpm / 60. * 2 * M_PI);//max joint speed in rad/s

	//sim.setViewport(Vec3d(-0.3, 0, 0.3), Vec3d(0, 0, 0), Vec3d(0, 0, 1));
	//sim.setViewport(Vec3d(0.6, 0, 0.3), Vec3d(0, 0, 0.2), Vec3d(0, 0, 1));
	sim.setViewport(Vec3d(1.75, -2.5, 1.0), Vec3d(1.75, 0, 0.1), Vec3d(0, 0, 1));


	// our plane has a unit normal in the z-direction, with 0 offset.
	//sim.createPlane(Vec3d(0, 0, 1), -1, 0, 0);

	sim.global_acc = Vec3d(0, 0, -9.8); // global acceleration
	sim.createPlane(Vec3d(0, 0, 1), 0, 0.6, 0.6);
	//sim.createPlane(Vec3d(0, 0, 1), 0, 0.4, 0.35);


	double runtime = 1600;
	sim.setBreakpoint(runtime);

	sim.start();
	//sim.pause(1);
	//while (sim.RUNNING) {
	//	std::this_thread::sleep_for(std::chrono::milliseconds(1));
	//}
	//sim.resume();

	auto end = std::chrono::steady_clock::now();
	printf("main():Elapsed time:%d ms \n",
		(int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

	return 0;
}

