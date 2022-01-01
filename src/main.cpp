#define _USE_MATH_DEFINES
#include <math.h>
#include <omp.h>
#include <thread>
#include <complex>
#include <chrono> // for time measurement
#include<algorithm>

#include <msgpack.hpp>
#include <clipp.h>
#include <filesystem>

#include "sim.h"

namespace fs = std::filesystem;

void keyboardCallback(Simulation* sim);

int main(int argc, char* argv[])
{
	// for time measurement
	auto start = std::chrono::steady_clock::now();

	int device = 0;// cuda device
	bool help = 1;
	std::string model_path_str = getProgramDir() + "//flexipod_12dof.msgpack";
	{	//https://github.com/muellan/clipp
		using namespace clipp;
		auto cli = (
			(
				option("-d", "--device").doc("set cuda device") & value("cuda device#", device),
				option("-m", "--model").doc("set model path") & value("model path", model_path_str)
				)
			| option("-h", "--help").doc("show help").set(help, true)
			);
		if (!parse(argc, argv, cli) || help) {
			std::cout << make_man_page(cli, argv[0]);
			//return 0;
		}
	}
	fs::path model_path(model_path_str);
	if (!fs::exists(model_path)) {// make sure the file exists
		std::cout << model_path << " does not exist! \n";
		exit(-1);
	}

	std::cout << device << "," << model_path_str << std::endl;


	gpuErrchk(cudaSetDevice(device)); // set cuda device

	std::cout << "current working dir: " << getWorkingDir() << "\n";
	std::cout << "program dir: " << getProgramDir() << "\n";

	Model bot(model_path_str); //defined in sim.h


	//for (const auto& i:bot.id_vertices["part"])
	//	std::cout << i << ' ';

	const size_t num_body = bot.id_vertices.at("part").size() - 1;//number of bodies
	const size_t num_mass = bot.vertices.size(); // number of mass
	const size_t num_spring = bot.edges.size(); // number of spring
	const size_t num_triangle = bot.triangles.size();// number of triangle
	const size_t num_joint = bot.joints.size();//number of rotational joint

	Simulation sim(num_mass, num_spring,num_joint,num_triangle,device); // Simulation object

	MASS& mass = sim.mass; // reference variable for sim.mass
	SPRING& spring = sim.spring; // reference variable for sim.spring
	TRIANGLE triangle = sim.triangle;// reference variable for sim.triangle






	

	
	//sim.pause();

	//constexpr double radius_poisson = 12.5 * 1e-3;
	const double  radius_poisson = bot.radius_poisson;
	const double radius_knn = radius_poisson * sqrt(3.0);
	const double min_radius = radius_poisson * 0.5;

	const double m = 0.09* radius_poisson;// mass per vertex
	const double inv_m = 1. / m;

	sim.dt = 6e-5; // timestep
	const double spring_compliance = 1 / (m * 6e6); //spring constant for silicone leg [m/N]
	const double spring_damping = m * 6e4; // damping for spring [N*s/m]
	sim.USE_PBD = true;

	//sim.dt = 3e-5; // timestep
	//const double spring_compliance = 1 / (m * 6e6); //spring constant for silicone leg [m/N]
	//const double spring_damping = m * 1e3; // damping for spring [N*s/m]
	//sim.USE_PBD = false;

	constexpr double scale_rigid = 3.0;// scaling factor rigid
	constexpr double scale_soft = 2.0; // scaling factor soft
	constexpr double scale_rot_spring = 1.0; // stiffness scale for rotation spring

	constexpr double scale_joint_m = 2.6; // scaling factor for the joint mass
	constexpr double scale_joint_compliance = 3; // scaling factor for the joint spring constant
	constexpr double scale_joint_damping = 3; // scaling factor for the joint spring damping

	const double inv_m_joint = inv_m / scale_joint_m; // inverse joint mass

	//const double scale_low = 0.5; // scaling factor low
	constexpr double scale_probe = 0.08; // scaling factor for the probs, e.g. coordinates

	const double spring_compliance_rigid = spring_compliance / scale_rigid;//spring constant for rigid spring
	const double spring_compliance_soft = spring_compliance / scale_soft;//spring constant for soft spring

	const double spring_compliance_restable = spring_compliance / scale_joint_compliance; // spring constant for resetable spring
	const double spring_damping_restable = spring_damping * scale_joint_damping; // spring damping for resetable spring

	//const double spring_compliance_restable = 0; // spring constant for resetable spring
	//const double spring_damping_restable = 0; // spring damping for resetable spring

	// spring coefficient for the probing springs, e.g. coordinates
	const double inv_m_probe = 1./(m * scale_probe);//mass for the probe
	const double spring_compliance_probe_anchor = spring_compliance / (scale_probe*2.0); // spring constant for coordiates anchor springs
	const double spring_compliance_probe_self = spring_compliance / (scale_probe*2.0); // spring constant for coordiates self springs
	const double spring_damping_probe = spring_damping * scale_probe * 2.0;

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
		mass.pos_prev[i] = mass.pos[i];// previous pos
		mass.color[i] = bot.colors[i]; // color (Vec3d) [0.0-1.0]
		mass.inv_m[i] = inv_m; // mass [kg]
		// conditionally apply constraint to the masses
		mass.flag[i].assignBit(MASS_FLAG_CONSTRAIN_ID, bot.is_surface[i]);
	}
#pragma omp simd
	for (int i = 0; i < num_spring; i++)
	{
		spring.edge[i] = bot.edges[i]; // the (left,right) mass index of the spring
		spring.damping[i] = spring_damping; // spring constant
		spring.rest[i] = (mass.pos[spring.edge[i].x] - mass.pos[spring.edge[i].y]).norm(); // spring rest length
		// longer spring will have a smalller influence, https://ccrma.stanford.edu/~jos/pasp/Young_s_Modulus_Spring_Constant.html
		spring.compliance[i] = spring_compliance_soft / radius_knn * std::max(spring.rest[i], min_radius); // spring constant
		spring.resetable[i] = false; // set all spring as non-resetable
	}

	// get internal vertices (vertices that inside body parts)
	std::vector<bool> internal_vert(num_mass,true); // bool of internal vertices
	for (int i = bot.id_vertices.at("part").back(); i < num_mass; i++)
		internal_vert[i] = false; // excluding anchor and coord..
	for each (auto tri in bot.triangles)// excluding surfaces
	{	
		internal_vert[tri.x] = false;
		internal_vert[tri.y] = false;
		internal_vert[tri.z] = false;
	}
	//double scale_internal_spring = scale_soft/4.0; // multiplier for internal spring compliance
	//for (int i = 0; i < num_spring; i++) { // using rigid springs for internal constructions
	//	auto e = spring.edge[i];
	//	if ((internal_vert[e.x] && internal_vert[e.x])) {
	//		spring.compliance[i] = spring.compliance[i] * scale_internal_spring;
	//	}
	//}


	 //// set higher mass value for robot body
	 //for (int i = bot.idVertices[0]; i < bot.idVertices[1]; i++)
	 //{
	 //	mass.inv_m[i] = 1./(m*1.8); // accounting for addional mass for electornics
	 //}
	 //// set lower mass value for leg
	 //for (int i = bot.idVertices[1]; i < bot.idVertices[num_body]; i++)
	 //{
	 //	mass.inv_m[i] = 1./(m * 0.3); // 80% infill,no skin
	 //}

	 // set the mass value for joint
#pragma omp simd
	for (int i = 0; i < bot.joints.size(); i++)
	{
		for each (int j in bot.joints[i].left)
		{
			mass.inv_m[j] = inv_m_joint;
		}
		for each (int j in bot.joints[i].right)
		{
			mass.inv_m[j] = inv_m_joint;
		}
	}

	// set higher spring constant for the rotational joints
	for (int i = bot.id_edges.at("anchor").front(); i < bot.id_edges.at("anchor").back(); i++)
	{
		spring.compliance[i] = spring_compliance_rigid; // joints anchors
		//spring.compliance[i] = 0; // joints anchors
		//spring.damping[i] = 0;
	}

	for (int i = bot.id_edges.at("rot_spring").front(); i < bot.id_edges.at("rot_spring").back(); i++)
	{
		spring.compliance[i] = spring_compliance/scale_rot_spring; // joints rotation spring
		spring.damping[i] = spring.damping[i] * scale_rot_spring; // joints rotation spring
		//spring.compliance[i] =0; // joints rotation spring
		//spring.damping[i] = 0;
	}


	sim.id_oxyz_start = bot.id_vertices.at("part_coord").front();
	sim.id_oxyz_end = bot.id_vertices.at("joint_coord").back();

	// set lower mass for the anchored coordinate systems
	for (int i = sim.id_oxyz_start; i < sim.id_oxyz_end; i++)
	{
		mass.inv_m[i] = inv_m_probe; // mass [kg]
	}

	for (int i = bot.id_edges.at("coord").front(); i < bot.id_edges.at("coord").back(); i++)
	{
		spring.compliance[i] = spring_compliance_probe_self;// coordinate_self_springs
		spring.damping[i] = spring_damping_probe;
	}
	for (int i = bot.id_edges.at("coord_attach").front(); i < bot.id_edges.at("coord_attach").back(); i++)
	{
		spring.compliance[i] = spring_compliance_probe_anchor;// coordinate_anchor_springs
		spring.damping[i] = spring_damping_probe;
	}

	sim.joint.init(bot, true);
	sim.d_joint.init(bot, false);

	//
	//for (int i = 0; i < length; i++)
	//{

	//}
	//id_vertices["part"].front()


	auto& joint = sim.joint; // set spring joint_id
	for (int i = 0; i < joint.edge_num; i++) {
		int jid = joint.edge_id[i];
		spring.compliance[jid] = spring_compliance_restable;// resetable spring, reset the rest length per dynamic update
		spring.damping[jid] = spring_damping_restable;
		spring.resetable[jid] = true;
		spring.joint_id[jid] = i;
	}


	// print out mass statistics
	double total_mass = 0;
//#pragma omp parallel for shared (total_mass,mass.inv_m) reduction(+:total_mass)
	for (int i = 0; i < num_mass; i++) { total_mass += 1./mass.inv_m[i]; }
	printf("total mass:%.2f [kg]\n",total_mass);

	std::vector<double> part_mass(num_body,0); // vector of size num_body, with values 0
#pragma omp parallel for
	for (int k = 0; k < num_body; k++)
		for (int i = bot.id_vertices.at("part")[k]; i < bot.id_vertices.at("part")[k + 1]; i++)
			part_mass[k] += 1./mass.inv_m[i];
	printf("part mass:");
	for (const auto& i : part_mass) 
		printf("%.2f ", i);
	printf("[kg]\n");

	std::vector<double> joint_mass(num_joint, 0); // vector of size num_body, with values 0
#pragma omp parallel for
	for (int k = 0; k < num_joint; k++)
		for (const auto & i:bot.joints[k].left)
			joint_mass[k] += 1./mass.inv_m[i];
	printf("joint mass:");
	for (const auto& i : joint_mass)
		printf("%.2f ", i);
	printf("[kg]\n");


	// set max speed for each joint
	double max_rpm = 300;//maximun revolution per minute
	sim.joint_control.max_vel = max_rpm / 60. * 2 * M_PI;//max joint speed in rad/s
	double settling_time = 1.0;// reaches max_rpm within this time
	sim.joint_control.max_acc = sim.joint_control.max_vel / settling_time;

#ifdef GRAPHICS
	//sim.setViewport(Vec3d(-0.3, 0, 0.3), Vec3d(0, 0, 0), Vec3d(0, 0, 1));
	//sim.setViewport(Vec3d(0.6, 0, 0.3), Vec3d(0, 0, 0.2), Vec3d(0, 0, 1));
	sim.setViewport(glm::vec3(1, -0, 1.0), glm::vec3(0, 0, 0.4), glm::vec3(0, 0, 1));
#endif // GRAPHICS

	// our plane has a unit normal in the z-direction, with 0 offset.
	//sim.createPlane(Vec3d(0, 0, 1), -1, 0, 0);

	sim.global_acc = Vec3d(0, 0, -9.8); // global acceleration

	sim.createPlane(Vec3d(0, 0, 1), 0, 0.8, 0.7,0.5,20);
	//sim.createPlane(Vec3d(0, 0, 1), -1, 0.8, 0.7);

	//// fix the main body
	//for (int i = bot.id_vertices.at("part").at(0); i < bot.id_vertices.at("part").at(1); i++) {
	//	mass.flag[i].assignBit(MASS_FLAG_DOF_X_ID, false);
	//	mass.flag[i].assignBit(MASS_FLAG_DOF_Y_ID, false);
	//	//mass.flag[i].assignBit(MASS_FLAG_DOF_Z_ID, false);
	//}

	//sim.createPlane(Vec3d(0, 0, 1), 0, 0.8, 0.7);
	//sim.createPlane(Vec3d(0, 0, -1), -0.6, 0.9, 0.8,0.05f,0.1f);

	//sim.createBall(Vec3d(-0.09, 0, 0.5), 0.05);
	//sim.createBall(Vec3d(0.09, 0, 0.5), 0.05);

	//sim.createPlane(Vec3d(0, 0, 1), 0, 0.4, 0.35);


	//double runtime = 172800;//48 hours
	//double runtime = 15;//48 hours
	//sim.setBreakpoint(runtime,true);


	
#ifdef MEASURE_CONSTRAINT
	sim.id_vertices = bot.id_vertices;
	sim.num_body = num_body;
	sim.body_constraint_force.resize(num_body);
	sim.total_mass = total_mass;
#endif //MEASURE_CONSTRAINT

#ifdef STRESS_TEST
		sim.id_selected_edges = bot.id_selected_edges;
#endif


#ifdef UDP
		sim.udp_delay_step = 0;//offset by x timesteps
		double udp_delay_ms = sim.udp_delay_step * sim.dt * sim.NUM_QUEUED_KERNELS * 1000;
		printf("UDP is delayed by %d step (%.3f ms) \n", sim.udp_delay_step, udp_delay_ms);
#endif // UDP

		sim.keyboardCallback = keyboardCallback;

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


void keyboardCallback(Simulation* sim) {
	constexpr double speed_multiplier = 0.2;
	constexpr double pos_multiplier = 0.1;

	if (sim->getKey(GLFW_KEY_UP)) {
		for (int i = 0; i < sim->joint.size(); i++) {
			if (sim->joint_control.mode == JointControlMode::vel) {
				sim->joint_control.vel_desired[i] += i < 2 ? speed_multiplier : -speed_multiplier;
			}
			else if (sim->joint_control.mode == JointControlMode::pos) {
				sim->joint_control.pos_desired[i] += i < 2 ? pos_multiplier : -pos_multiplier;
			}
		}
	}
	else if (sim->getKey(GLFW_KEY_DOWN)) {
		for (int i = 0; i < sim->joint.size(); i++) {
			if (sim->joint_control.mode == JointControlMode::vel) {
				sim->joint_control.vel_desired[i] -= i < 2 ? speed_multiplier : -speed_multiplier;
			}
			else if (sim->joint_control.mode == JointControlMode::pos) {
				sim->joint_control.pos_desired[i] -= i < 2 ? pos_multiplier : -pos_multiplier;
			}
		}
	}
	if (sim->getKey(GLFW_KEY_LEFT)) {
		for (int i = 0; i < sim->joint.size(); i++) {
			if (sim->joint_control.mode == JointControlMode::vel) {
				sim->joint_control.vel_desired[i] -= speed_multiplier;
			}
			else if (sim->joint_control.mode == JointControlMode::pos) {
				sim->joint_control.pos_desired[i] -= pos_multiplier;
			}
		}
	}
	else if (sim->getKey(GLFW_KEY_RIGHT)) {
		for (int i = 0; i < sim->joint.size(); i++) {
			if (sim->joint_control.mode == JointControlMode::vel) {
				sim->joint_control.vel_desired[i] += speed_multiplier;
			}
			else if (sim->joint_control.mode == JointControlMode::pos) {
				sim->joint_control.pos_desired[i] += pos_multiplier;
			}
		}
	}
	else if (sim->getKey(GLFW_KEY_0)) { // zero speed
		sim->joint_control.reset(sim->mass, sim->joint);
	}
}