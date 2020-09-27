/*modified from the orginal Titan simulation libaray:https://github.com/jacobaustin123/Titan
ref: J. Austin, R. Corrales-Fatou, S. Wyetzner, and H. Lipson, �Titan: A Parallel Asynchronous Library for Multi-Agent and Soft-Body Robotics using NVIDIA CUDA,� ICRA 2020, May 2020.
*/

#define GLM_FORCE_PURE
#include "sim.h"


#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_gl_interop.h>
#include <exception>
#include <device_launch_parameters.h>
//#include <cooperative_groups.h>


constexpr int MAX_BLOCKS = 65535; // max number of CUDA blocks
constexpr int THREADS_PER_BLOCK = 64;
constexpr int MASS_THREADS_PER_BLOCK = 128;

constexpr int  NUM_QUEUED_KERNELS = 40; // number of kernels to queue at a given time (this will reduce the frequency of updates from the CPU by this factor
constexpr int NUM_UPDATE_PER_ROTATION = 4; //number of update per rotation


__global__ void SpringUpate(
	const MASS mass,
	const SPRING spring
) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < spring.num) {
		Vec2i e = spring.edge[i];
		Vec3d s_vec = mass.pos[e.y] - mass.pos[e.x];// the vector from left to right
		double length = s_vec.norm(); // current spring length

		s_vec /= (length > 1e-12 ? length : 1e-12);// normalized to unit vector (direction), check instablility for small length

		Vec3d force = spring.k[i] * (spring.rest[i] - length) * s_vec; // normal spring force
		force += s_vec.dot(mass.vel[e.x] - mass.vel[e.y]) * spring.damping[i] * s_vec;// damping

		mass.force[e.y].atomicVecAdd(force); // need atomics here
		mass.force[e.x].atomicVecAdd(-force); // removed condition on fixed

//#ifdef ROTATION
//		if (spring.resetable[i]) {
//			spring.rest[i] = length;//reset the spring rest length if this spring is restable
//		}
//#endif // ROTATION
	}

}

__global__ void SpringUpateReset(
	const MASS mass,
	const SPRING spring
) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < spring.num) {
		Vec2i e = spring.edge[i];
		Vec3d s_vec = mass.pos[e.y] - mass.pos[e.x];// the vector from left to right
		double length = s_vec.norm(); // current spring length
		s_vec /= (length > 1e-12 ? length : 1e-12);// normalized to unit vector (direction), check instablility for small length

		Vec3d force = spring.k[i] * (spring.rest[i] - length) * s_vec; // normal spring force
		force += s_vec.dot(mass.vel[e.x] - mass.vel[e.y]) * spring.damping[i] * s_vec;// damping

		mass.force[e.y].atomicVecAdd(force); // need atomics here
		mass.force[e.x].atomicVecAdd(-force); // removed condition on fixed

#ifdef ROTATION
		if (spring.resetable[i]) {
			spring.rest[i] = length;//reset the spring rest length if this spring is restable
		}
#endif // ROTATION
	}

}


__global__ void MassUpate(
	const MASS mass,
	const CUDA_GLOBAL_CONSTRAINTS c,
	const Vec3d global_acc,
	const double dt) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < mass.num) {
		if (mass.fixed[i] == false) {
			double m = mass.m[i];
			Vec3d pos = mass.pos[i];
			Vec3d vel = mass.vel[i];

			Vec3d force = mass.force[i];
			force += mass.force_extern[i];// add spring force and external force [N]

			/*if (mass.constrain)*/ {
				for (int j = 0; j < c.num_planes; j++) { // global constraints
					c.d_planes[j].applyForce(force, pos, vel); // todo fix this 
				}
				for (int j = 0; j < c.num_balls; j++) {
					c.d_balls[j].applyForce(force, pos);
				}
			}

// euler integration
			force /= m;// force is now acceleration
			force += global_acc;// add global accleration
			vel += force * dt; // vel += acc*dt
			mass.acc[i] = force; // update acceleration
			mass.vel[i] = vel; // update velocity
			mass.pos[i] += vel * dt; // update position
			mass.force[i].setZero();

		}
	}
}

#ifdef ROTATION
__global__ void massUpdateAndRotate(
	const MASS mass,
	const CUDA_GLOBAL_CONSTRAINTS c,
	const JOINT joint,
	const Vec3d global_acc,
	const double dt) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < mass.num) {
		if (mass.fixed[i] == false) {
			double m = mass.m[i];
			Vec3d pos = mass.pos[i];
			Vec3d vel = mass.vel[i];

			Vec3d force = mass.force[i];
			force += mass.force_extern[i];// add spring force and external force [N]

			/*if (mass.constrain)*/ {
				for (int j = 0; j < c.num_planes; j++) { // global constraints
					c.d_planes[j].applyForce(force, pos, vel); // todo fix this 
				}
				for (int j = 0; j < c.num_balls; j++) {
					c.d_balls[j].applyForce(force, pos);
				}
			}

// euler integration
			force /= m;// force is now acceleration
			force += global_acc;// add global accleration
			vel += force * dt; // vel += acc*dt
			mass.acc[i] = force; // update acceleration
			mass.vel[i] = vel; // update velocity
			mass.pos[i] += vel * dt; // update position
			mass.force[i].setZero(); // reset force
		}
	}
	else if ((i -= mass.num) < joint.points.num) {// this part is same as rotateJoint
		if (i < joint.anchors.num) {
			Vec2i e = joint.anchors.edge[i];
			joint.anchors.dir[i] = (mass.pos[e.y] - mass.pos[e.x]).normalize();
		}
		__threadfence();
		__syncthreads();

		int anchor_id = joint.points.anchorId[i];
		int mass_id = joint.points.massId[i];

		mass.pos[mass_id] = AxisAngleRotaion(joint.anchors.dir[anchor_id], mass.pos[mass_id],
			joint.anchors.theta[anchor_id] * joint.points.dir[i], mass.pos[joint.anchors.edge[anchor_id].x]);
	}
}


//__global__ void rotateJoint(Vec3d* __restrict__ mass_pos, const JOINT joint) {
//	int i = blockIdx.x * blockDim.x + threadIdx.x;
//	if (i < joint.points.num) {
//		if (i < joint.anchors.num) {
//			Vec2i e = joint.anchors.edge[i];
//			joint.anchors.dir[i] = (mass_pos[e.y] - mass_pos[e.x]).normalize();
//		}
//		__threadfence();
//		__syncthreads();
//
//		int anchor_id = joint.points.anchorId[i];
//		int mass_id = joint.points.massId[i];
//
//		mass_pos[mass_id] = AxisAngleRotaion(joint.anchors.dir[anchor_id], mass_pos[mass_id],
//			joint.anchors.theta[anchor_id] * joint.points.dir[i], mass_pos[joint.anchors.edge[anchor_id].x]);
//
//		//Vec3d anchor_left = mass_pos[joint.anchors.left[anchor_id]];
//		//Vec3d anchor_right = mass_pos[joint.anchors.right[anchor_id]];
//		////Vec3d dir = 
//		///*double3*/
//		//Vec3d dir = (mass_pos[joint.anchors.right[anchor_id]] - mass_pos[joint.anchors.left[anchor_id]]).normalize();
//
//		//mass_pos[mass_id] = AxisAngleRotaion(dir, mass_pos[mass_id],
//		//	joint.anchors.theta[anchor_id] * joint.points.dir[i], mass_pos[joint.anchors.left[anchor_id]]);
//	}
//}

__global__ void rotateJoint(Vec3d* __restrict__ mass_pos, const JOINT joint) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < joint.points.num) {

		int anchor_id = joint.points.anchorId[i];
		int mass_id = joint.points.massId[i];
		Vec2i anchor_edge = joint.anchors.edge[anchor_id]; // mass id of the achor edge point
		mass_pos[mass_id] = AxisAngleRotaion(
			mass_pos[anchor_edge.x],
			mass_pos[anchor_edge.y], mass_pos[mass_id],
			joint.anchors.theta[anchor_id] * joint.points.dir[i]);

		//Vec3d anchor_left = mass_pos[joint.anchors.left[anchor_id]];
		//Vec3d anchor_right = mass_pos[joint.anchors.right[anchor_id]];
		////Vec3d dir = 
		///*double3*/
		//Vec3d dir = (mass_pos[joint.anchors.right[anchor_id]] - mass_pos[joint.anchors.left[anchor_id]]).normalize();

		//mass_pos[mass_id] = AxisAngleRotaion(dir, mass_pos[mass_id],
		//	joint.anchors.theta[anchor_id] * joint.points.dir[i], mass_pos[joint.anchors.left[anchor_id]]);
	}
}

#endif // ROTATION


Simulation::Simulation() {
	//dynamicsUpdate(d_mass.m, d_mass.pos, d_mass.vel, d_mass.acc, d_mass.force, d_mass.force_extern, d_mass.fixed,
	//	d_spring.k,d_spring.rest,d_spring.damping,d_spring.left,d_spring.right,
	//	d_mass.num,d_spring.num,global_acc, d_constraints,dt);

	for (int i = 0; i < NUM_CUDA_STREAM; ++i)
		cudaStreamCreate(&stream[i]);// create extra cuda stream: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution

}

Simulation::Simulation(size_t num_mass, size_t num_spring):Simulation() {
	mass = MASS(num_mass, true); // allocate host
	d_mass = MASS(num_mass, false); // allocate device
	spring = SPRING(num_spring, true); // allocate host
	d_spring = SPRING(num_spring, false); // allocate device
	//this->num_mass = num_mass;//refer to spring.num
	//this->num_spring = num_spring;// refer to mass.num
	//cudaDeviceSynchronize();
}

void Simulation::getAll() {//copy from gpu
	mass.copyFrom(d_mass, stream[NUM_CUDA_STREAM - 1]); // mass
	spring.copyFrom(d_spring, stream[NUM_CUDA_STREAM - 1]);// spring
	//cudaDeviceSynchronize();
}

void Simulation::setAll() {//copy form cpu
	d_mass.copyFrom(mass, stream[NUM_CUDA_STREAM - 1]);
	d_spring.copyFrom(spring, stream[NUM_CUDA_STREAM - 1]);
	//cudaDeviceSynchronize();
}

void Simulation::setMass() {
	d_mass.copyFrom(mass, stream[NUM_CUDA_STREAM - 1]);
}

inline int Simulation::computeBlocksPerGrid(const int threadsPerBlock, const int num) {
	int blocksPerGrid = (num - 1 + threadsPerBlock) / threadsPerBlock;
	if (blocksPerGrid > MAX_BLOCKS) { blocksPerGrid = MAX_BLOCKS; }
	return blocksPerGrid;
}

inline void Simulation::updateCudaParameters() {
	massBlocksPerGrid = computeBlocksPerGrid(MASS_THREADS_PER_BLOCK, mass.num);
	springBlocksPerGrid = computeBlocksPerGrid(THREADS_PER_BLOCK, spring.num);
	jointBlocksPerGrid = computeBlocksPerGrid(MASS_THREADS_PER_BLOCK, d_joint.points.num);
}

void Simulation::setBreakpoint(const double time) {
	if (ENDED) { throw std::runtime_error("Simulation has ended. Can't modify simulation after simulation end."); }
	bpts.insert(time); // TODO mutex breakpoints
}

/*pause the simulation at (simulation) time t [s] */
void Simulation::pause(const double t) {
	if (ENDED && !FREED) { throw std::runtime_error("Simulation has ended. can't call control functions."); }
	setBreakpoint(t);
	waitForEvent();
}

void Simulation::resume() {
	if (ENDED) { throw std::runtime_error("Simulation has ended. Cannot resume simulation."); }
	if (!STARTED) { throw std::runtime_error("Simulation has not started. Cannot resume before calling sim.start()."); }
	if (mass.num == 0) { throw std::runtime_error("No masses have been added. Add masses before simulation starts."); }
	updateCudaParameters();
	cudaDeviceSynchronize();
	RUNNING = true;
}

void Simulation::waitForEvent() {
	if (ENDED && !FREED) { throw std::runtime_error("Simulation has ended. can't call control functions."); }
	while (RUNNING) {
		std::this_thread::sleep_for(std::chrono::nanoseconds(100));
	}
}


/*backup the robot mass/spring/joint state */
void Simulation::backupState() {
	backup_spring = SPRING(spring, true);
	backup_mass = MASS(mass, true);
	backup_joint = JOINT(joint, true);
}
/*restore the robot mass/spring/joint state to the backedup state *///TODO check if other variable needs resetting
void Simulation::resetState() {//TODO...fix bug
	
	d_mass.copyFrom(backup_mass, stream[NUM_CUDA_STREAM - 1]);
	d_spring.copyFrom(backup_spring, stream[NUM_CUDA_STREAM - 1]);
	d_joint.copyFrom(backup_joint, stream[NUM_CUDA_STREAM - 1]);
	//size_t nbytes = joint.size() * sizeof(double);
	//memset(joint_speeds_cmd, 0, nbytes);
	//memset(joint_speeds, 0, nbytes);
	//memset(joint_angles, 0, nbytes);
	//memset(joint_speeds_desired, 0, nbytes);
	//memset(joint_speeds_error, 0, nbytes);
	//memset(joint_speeds_error_integral, 0, nbytes);
	for (int i = 0; i < joint.size(); i++) { 
		joint_speeds_cmd[i] = 0.; 
		joint_speeds[i] = 0.;
		joint_angles[i] = 0.;
		joint_speeds_desired[i] = 0.;
		joint_speeds_error[i] = 0.;
		joint_speeds_error_integral[i] = 0.;

	}

}


void Simulation::start() {
	if (ENDED) { throw std::runtime_error("The simulation has ended. Cannot call sim.start() after the end of the simulation."); }
	if (mass.num == 0) { throw std::runtime_error("No masses have been added. Please add masses before starting the simulation."); }
	printf("Starting simulation with %d masses and %d springs\n", mass.num, spring.num);
	RUNNING = true;
	STARTED = true;

	T = 0;

	if (this->dt == 0.0) { // if dt hasn't been set by the user.
		dt = 0.01; // min delta
	}
	updateCudaParameters();

	d_constraints.d_balls = thrust::raw_pointer_cast(&d_balls[0]);
	d_constraints.d_planes = thrust::raw_pointer_cast(&d_planes[0]);
	d_constraints.num_balls = d_balls.size();
	d_constraints.num_planes = d_planes.size();

	update_constraints = false;

	cudaMallocHost((void**)&joint_speeds_error_integral, joint.size() * sizeof(double));//initialize joint speed error integral array 
	cudaMallocHost((void**)&joint_speeds_error, joint.size() * sizeof(double));//initialize joint speed error array 
	cudaMallocHost((void**)&joint_speeds_cmd, joint.size() * sizeof(double));//initialize joint speed (commended) array 
	cudaMallocHost((void**)&joint_speeds_desired, joint.size() * sizeof(double));//initialize joint speed (desired) array 
	cudaMallocHost((void**)&joint_speeds, joint.size() * sizeof(double));//initialize joint speed (measured) array 
	cudaMallocHost((void**)&joint_angles, joint.size() * sizeof(double));//initialize joint angle (measured) array 

	setAll();// copy mass and spring to gpu

	backupState();// backup the robot mass/spring/joint state

#ifdef UDP
//Todo
	udp_server.run();

#endif //UDP


	gpu_thread = std::thread(&Simulation::_run, this); //TODO: thread
}

void Simulation::_run() { // repeatedly start next

	//cudaDeviceProp deviceProp;
	//cudaGetDeviceProperties(&deviceProp, 0);
	//if (deviceProp.cooperativeLaunch == 0) {
	//	printf("not supported");
	//	exit(-1);
	//}

#ifdef GRAPHICS
	createGLFWWindow(); // create a window with  width and height
	glGenVertexArrays(1, &VertexArrayID);//GLuint VertexArrayID;
	glBindVertexArray(VertexArrayID);

	// Create and compile our GLSL program from the shaders
	this->programID = LoadShaders(); // ("shaders/StandardShading.vertexshader", "shaders/StandardShading.fragmentshader"); //
	// Get a handle for our "MVP" uniform
	computeMVP(); // compute perspective projection matrix
	this->MatrixID = glGetUniformLocation(programID, "MVP"); // doesn't seem to be necessary
	generateBuffers(); // generate buffers for all masses and springs

	for (Constraint* c : constraints) { // generate buffers for constraint objects
		c->generateBuffers();
	}

	glUseProgram(programID);// Use our shader

	//updateBuffers();//Todo might not need?
#endif
	execute();
	GPU_DONE = true;
}


void Simulation::execute() {
	cudaDeviceSynchronize();//sync before while loop
	auto start = std::chrono::steady_clock::now();

#ifdef DEBUG_ENERGY
	energy_start = energy(); // compute the total energy of the system at T=0
#endif // DEBUG_ENERGY


	while (true) {
		if (!bpts.empty() && *bpts.begin() <= T) {
			cudaDeviceSynchronize(); // synchronize before updating the springs and mass positions
		//            std::cout << "Breakpoint set for time " << *bpts.begin() << " reached at simulation time " << T << "!" << std::endl;
			bpts.erase(bpts.begin());
			RUNNING = false;
			while (!RUNNING) {
				std::this_thread::sleep_for(std::chrono::nanoseconds(100));
				if (ENDED) {

					auto end = std::chrono::steady_clock::now();
					double duration = (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.;//[seconds]
					double sim_time_ratio = T / duration;
					double spring_update_rate = ((double)spring.num) / dt * sim_time_ratio;

					printf("Elapsed time:%.2f s for %.2f simulation time (%.2f); # %.2e spring update/s\n",
						duration, T, sim_time_ratio, spring_update_rate);

					//for (Constraint* c : constraints) {
					//	delete c;
					//}
#ifdef GRAPHICS

					deleteVBO(&vbo_vertex, cuda_resource_vertex);
					deleteVBO(&vbo_color, cuda_resource_color);
					deleteVBO(&vbo_edge, cuda_resource_edge);

					glDeleteProgram(programID);
					glDeleteVertexArrays(1, &VertexArrayID);
					glfwTerminate(); // Close OpenGL window and terminate GLFW
#endif
					return;
				}
			}
			if (resize_buffers) {
				resizeBuffers(); // needs to be run from GPU thread
				resize_buffers = false;
				update_colors = true;
				update_indices = true;
			}

			if (update_constraints) {
				d_constraints.d_balls = thrust::raw_pointer_cast(&d_balls[0]);
				d_constraints.d_planes = thrust::raw_pointer_cast(&d_planes[0]);
				d_constraints.num_balls = d_balls.size();
				d_constraints.num_planes = d_planes.size();

				for (Constraint* c : constraints) { // generate buffers for constraint objects
					if (!c->_initialized)
						c->generateBuffers();
				}
				update_constraints = false;
			}
			continue;
		}

		////cudaDeviceSynchronize();
		//cudaEvent_t event; // an event that tel
		//cudaEventCreateWithFlags(&event, cudaEventDisableTiming); // create event,disable timing for faster speed:https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf
		//cudaEvent_t event_rotation;
		//cudaEventCreateWithFlags(&event_rotation, cudaEventDisableTiming);

		for (int i = 0; i < (NUM_QUEUED_KERNELS / NUM_UPDATE_PER_ROTATION); i++) {

			for (int j = 0; j < NUM_UPDATE_PER_ROTATION - 1; j++) {

				SpringUpate << <springBlocksPerGrid, THREADS_PER_BLOCK >> > (d_mass, d_spring);
				MassUpate << <massBlocksPerGrid, MASS_THREADS_PER_BLOCK >> > (d_mass, d_constraints, global_acc, dt);
				//gpuErrchk(cudaPeekAtLastError());
			}
			//cudaEventRecord(event, 0);
			//cudaStreamWaitEvent(stream[0], event, 0);
			//cudaEventRecord(event_rotation, stream[0]);
			//cudaStreamWaitEvent(NULL, event_rotation, 0);

#ifdef ROTATION
			rotateJoint << <jointBlocksPerGrid, MASS_THREADS_PER_BLOCK >> > (d_mass.pos, d_joint);
			SpringUpateReset << <springBlocksPerGrid, THREADS_PER_BLOCK >> > (d_mass, d_spring);
			MassUpate << <massBlocksPerGrid, MASS_THREADS_PER_BLOCK >> > (d_mass, d_constraints, global_acc, dt);

			//SpringUpate << <springBlocksPerGrid, THREADS_PER_BLOCK >> > (d_mass, d_spring);
			//massUpdateAndRotate << <massBlocksPerGrid + jointBlocksPerGrid, MASS_THREADS_PER_BLOCK >> > (d_mass, d_constraints, d_joint, global_acc, dt);
			//SpringUpateReset << <springBlocksPerGrid, THREADS_PER_BLOCK >> > (d_mass, d_spring);
			//MassUpate << <massBlocksPerGrid, MASS_THREADS_PER_BLOCK >> > (d_mass, d_constraints, global_acc, dt);

			//gpuErrchk(cudaPeekAtLastError());
#endif // ROTATION
		}
		//cudaEventDestroy(event);//destroy the event, necessary to prevent memory leak
		//cudaEventDestroy(event_rotation);//destroy the event, necessary to prevent memory leak

		T += NUM_QUEUED_KERNELS * dt;

		//if (fmod(T, 1. / 100.0) < NUM_QUEUED_KERNELS * dt) {
		mass.CopyPosVelAccFrom(d_mass, stream[NUM_CUDA_STREAM - 1]);
		cudaDeviceSynchronize();
		//#pragma omp parallel for
		for (int i = 0; i < joint.anchors.num; i++) // compute joint angles and angular velocity
		{
			Vec2i anchor_edge = joint.anchors.edge[i];
			Vec3d rotation_axis = (mass.pos[anchor_edge.y] - mass.pos[anchor_edge.x]).normalize();
			Vec3d x_left = mass.pos[joint.anchors.leftCoord[i] + 1] - mass.pos[joint.anchors.leftCoord[i]];//oxyz
			Vec3d x_right = mass.pos[joint.anchors.rightCoord[i] + 1] - mass.pos[joint.anchors.rightCoord[i]];//oxyz
			double angle = signedAngleBetween(x_left, x_right, rotation_axis); //joint angle in [-pi,pi]

			double delta_angle = angle - joint_angles[i];
			if (delta_angle > M_PI) {
				joint_speeds[i] = (delta_angle - 2 * M_PI) / (NUM_QUEUED_KERNELS * dt);
			}
			else if (delta_angle > -M_PI) {
				joint_speeds[i] = delta_angle / (NUM_QUEUED_KERNELS * dt);
			}
			else {
				joint_speeds[i] = (delta_angle + 2 * M_PI) / (NUM_QUEUED_KERNELS * dt);
			}
			joint_angles[i] = angle;

			///// <summary>
			///// keep the joint angle to 0 rad
			///// </summary>
			//joint_speeds_cmd[i] = -joint_angles[i]*50;
			//if (joint_speeds_cmd[i] > max_joint_speed) { joint_speeds_cmd[i] = max_joint_speed; }
			//if (joint_speeds_cmd[i] < -max_joint_speed) { joint_speeds_cmd[i] = -max_joint_speed; }
			//joint.anchors.theta[i] = NUM_UPDATE_PER_ROTATION * joint_speeds_cmd[i] * dt;// update joint speed
		}

		Vec3d com_pos = mass.pos[id_oxyz_start];//body center of mass position
		Vec3d com_acc = mass.acc[id_oxyz_start];//body center of mass acceleration

		Vec3d ox = (mass.pos[id_oxyz_start + 1] - com_pos).normalize();
		Vec3d oy = mass.pos[id_oxyz_start + 2] - com_pos;
		oy = (oy - oy.dot(ox) * ox).normalize();
#ifdef UDP
		msg_send.T = T;
		for (auto i = 0; i < 4; i++)
		{
			msg_send.jointAngle[i] = joint_angles[i];
			msg_send.jointSpeed[i] = joint_speeds[i];
		}

		for (auto i = 0; i < 3; i++)
		{
			msg_send.acceleration[i] = com_acc[i];
			msg_send.position[i] = com_pos[i];
			msg_send.orientation[i] = ox[i];
			msg_send.orientation[3 + i] = oy[i];
		}



		udp_server.msg_send = msg_send;
		udp_server.flag_should_send = true;
		// receiving message
		if (udp_server.flag_new_received) {
			msg_rec = udp_server.msg_rec;
			udp_server.flag_new_received = false;

			if (fmod(T, 1. / 10.0) < NUM_QUEUED_KERNELS * dt) {// print only once in a while
				printf("%3.3f \t %3.3f %3.3f %3.3f %3.3f\r\r", msg_rec.T,
					msg_rec.jointSpeed[0],
					msg_rec.jointSpeed[1],
					msg_rec.jointSpeed[2],
					msg_rec.jointSpeed[3]);
			}
			switch (msg_rec.header)
			{
			case UDP_HEADER::RESET: // reset
				// set the reset flag to true, resetState() will be called 
				// to restore the robot mass/spring/joint state to the backedup state
				RESET = true; 
				break;
			case UDP_HEADER::MOTOR_SPEED_COMMEND:
				for (int i = 0; i < joint.anchors.num; i++) {//update joint speed from received udp packet
					joint_speeds_desired[i] = msg_rec.jointSpeed[i];
				}
				break;
			default:
				break;
			}
		}

		if (fmod(T, 1. / 10.0) < NUM_QUEUED_KERNELS * dt) {
			//printf("% 6.1f",T); // time
			//printf("|t");
			//for (int i = 0; i < joint.anchors.num; i++) {
			//	printf("% 4.0f", joint_angles[i] * M_1_PI * 180.0); // display joint angle in deg
			//}
			//printf("|w");
			//for (int i = 0; i < joint.anchors.num; i++) {
			//	printf("% 4.0f", joint_speeds[i] * M_1_PI * 30.0); // display joint speed in RPM
			//}

			//printf("|wc");
			//for (int i = 0; i < joint.anchors.num; i++) {
			//	printf("% 4.0f", joint_speeds_cmd[i] * M_1_PI * 30.0); // display joint speed in RPM
			//}

			////printf("|xy%+ 2.2f %+ 2.2f %+ 2.2f ", ox.x, ox.y, ox.z);
			////printf("%+ 2.2f %+ 2.2f %+ 2.2f", oy.x, oy.y, oy.z);

			//printf("|x%+ 6.2f %+ 6.2f %+ 6.2f", com_pos.x, com_pos.y, com_pos.z);
			//printf("|a%+ 6.1f %+ 6.1f %+ 6.1f", com_acc.x, com_acc.y, com_acc.z);
			//printf("\r\r");
		}
		//printf("\n");
	//}
#endif // UDP

		for (int i = 0; i < joint.anchors.num; i++) // compute joint angles and angular velocity
		{// update joint_speeds_cmd

			joint_speeds_error[i] = joint_speeds_desired[i] - joint_speeds[i];

			joint_speeds_error_integral[i] += joint_speeds_error[i]; // simple proportional control
			if (joint_speeds_error_integral[i] > max_joint_speed) { joint_speeds_error_integral[i] = max_joint_speed; }
			if (joint_speeds_error_integral[i] < -max_joint_speed) { joint_speeds_error_integral[i] = -max_joint_speed; }

			joint_speeds_cmd[i] = 0.5 * joint_speeds_error[i] + 0.25 * joint_speeds_error_integral[i];

			if (joint_speeds_cmd[i] > max_joint_speed) { joint_speeds_cmd[i] = max_joint_speed; }
			if (joint_speeds_cmd[i] < -max_joint_speed) { joint_speeds_cmd[i] = -max_joint_speed; }
			//joint.anchors.theta[i] = 0.5 * NUM_UPDATE_PER_ROTATION * joint_speeds_cmd[i] * dt;// update joint speed
			joint.anchors.theta[i] = NUM_UPDATE_PER_ROTATION * joint_speeds_cmd[i] * dt;// update joint speed

		}
		// update joint speed
		d_joint.anchors.copyThetaFrom(joint.anchors, stream[0]);

#ifdef GRAPHICS
		if (fmod(T, 1. / 60.1) < NUM_QUEUED_KERNELS * dt) {

#ifdef DEBUG_ENERGY
			double e = energy();
			//if (abs(e - energy_start) > energy_deviation_max) {
			//	energy_deviation_max = abs(e - energy_start);
			//	printf("%.3f\t%.3f\n", T, energy_deviation_max / energy_start);
			//}
			//else { printf("%.3f\r", T); }
			printf("%.3f\t%.3f\n", T, e);

#endif // DEBUG_ENERGY

			Vec3d com_pos = mass.pos[id_oxyz_start];// center of mass position (anchored body center)

			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear screen

			if (glfwGetKey(window, GLFW_KEY_W)) {
				//camera_pos += 0.02 * (camera_dir - camera_dir.dot(camera_up) * camera_up);
				camera_h_offset -= 0.02;//move closer
			}
			else if (glfwGetKey(window, GLFW_KEY_S)) {
				//camera_pos -= 0.02 * (camera_dir - camera_dir.dot(camera_up) * camera_up);
				camera_h_offset += 0.02;//move away
			}

			if (glfwGetKey(window, GLFW_KEY_A)) {
				//camera_dir = AxisAngleRotaion(camera_up, camera_dir, 0.02, Vec3d());
				//camera_pos = AxisAngleRotaion(camera_up, camera_pos, 0.02, com_pos);
				camera_yaw -= 0.02;
			}
			else if (glfwGetKey(window, GLFW_KEY_D)) {
				//camera_dir = AxisAngleRotaion(camera_up, camera_dir, -0.02, Vec3d());
				//camera_pos = AxisAngleRotaion(camera_up, camera_pos, 0.02, com_pos);
				camera_yaw += 0.02;
			}
			else if (glfwGetKey(window, GLFW_KEY_Q)) {
				camera_up_offset -= 0.02;
			}
			else if (glfwGetKey(window, GLFW_KEY_E)) {
				camera_up_offset += 0.02;
			}

			double speed_multiplier = 0.1;
			//double speed_multiplier = 0.0001;
			if (glfwGetKey(window, GLFW_KEY_UP)) {
				if (joint_speeds_desired[0] < max_joint_speed) { joint_speeds_desired[0] += speed_multiplier; }
				if (joint_speeds_desired[1] < max_joint_speed) { joint_speeds_desired[1] += speed_multiplier; }
				if (joint_speeds_desired[2] > -max_joint_speed) { joint_speeds_desired[2] -= speed_multiplier; }
				if (joint_speeds_desired[3] > -max_joint_speed) { joint_speeds_desired[3] -= speed_multiplier; }
			}
			else if (glfwGetKey(window, GLFW_KEY_DOWN)) {
				if (joint_speeds_desired[0] > -max_joint_speed) { joint_speeds_desired[0] -= speed_multiplier; }
				if (joint_speeds_desired[1] > -max_joint_speed) { joint_speeds_desired[1] -= speed_multiplier; }
				if (joint_speeds_desired[2] < max_joint_speed) { joint_speeds_desired[2] += speed_multiplier; }
				if (joint_speeds_desired[3] < max_joint_speed) { joint_speeds_desired[3] += speed_multiplier; }
			}
			if (glfwGetKey(window, GLFW_KEY_LEFT)) {
				for (int i = 0; i < joint.size(); i++)
				{
					if (joint_speeds_desired[i] > -max_joint_speed) { joint_speeds_desired[i] -= speed_multiplier; }
				}
			}
			else if (glfwGetKey(window, GLFW_KEY_RIGHT)) {
				for (int i = 0; i < joint.size(); i++)
				{
					if (joint_speeds_desired[i] < max_joint_speed) { joint_speeds_desired[i] += speed_multiplier; }
				}
			}
			else if (glfwGetKey(window, GLFW_KEY_0)) { // zero speed
				for (int i = 0; i < joint.size(); i++) { joint_speeds_desired[i] = 0.; }
			}
			else if (glfwGetKey(window, GLFW_KEY_R)) { // reset
				RESET = true;
			}

			if (RESET) {
				RESET = false;
				resetState();// restore the robot mass/spring/joint state to the backedup state
			}

			// https://en.wikipedia.org/wiki/Slerp
			//mass.pos[id_oxyz_start].print();
			double t_lerp = 0.01;

			Vec3d camera_rotation_anchor = com_pos + (camera_up_offset-com_pos.dot(camera_up))*camera_up;

			Vec3d camera_pos_desired = AxisAngleRotaion(camera_up, camera_href_dir * camera_h_offset, camera_yaw, Vec3d())+ camera_rotation_anchor;

			camera_pos = lerp(camera_pos, camera_pos_desired, t_lerp);

			Vec3d camera_dir_desired = (com_pos - camera_pos).normalize();

			// spherical linear interpolation from camera_dir to camera_dir_new by factor t_lerp
			camera_dir = slerp(camera_dir, camera_dir_desired, t_lerp).normalize();

			computeMVP(true); // update MVP, also update camera matrix //todo

			glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);// update transformation "MVP" uniform


			for (Constraint* c : constraints) {
				c->draw();
			}

			updateBuffers();
			//updateVertexBuffers();

			//cudaDeviceSynchronize(); // synchronize before updating the springs and mass positions

			draw();

			// Swap buffers, render screen
			glfwPollEvents();
			glfwSwapBuffers(window);

			if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS || glfwWindowShouldClose(window) != 0) {
				bpts.insert(T);// break at current time T
				printf("\nwindow closed\n");
				//exit(0); // TODO maybe deal with memory leak here. //key press exit,
			}
		}
#endif //GRAPHICS
	}

}




Simulation::~Simulation() {
	std::cerr << "Simulation destructor called." << std::endl;

	if (STARTED) {
		waitForEvent();

		ENDED = true; // TODO maybe race condition

		while (!GPU_DONE) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));// TODO fix race condition
		}
		if (gpu_thread.joinable()) {

			gpu_thread.join();
		}
		else {
			printf("could not join GPU thread.\n");
			exit(1);
		}
#ifdef UDP
		udp_server.flag_should_close = true;
#endif // UDP

	}
	if (!FREED) {
		freeGPU();
	}
}


void Simulation::freeGPU() {
	FREED = true;
	ENDED = true; // just to be safe
	//Todo
	//for (Spring* s : springs) {
	//	s->_left = nullptr;
	//	s->_right = nullptr;
	//	delete s;
	//}
	//for (Mass* m : masses) {
	//	if (m->arrayptr) {
	//		gpuErrchk(cudaFree(m->arrayptr));
	//	}
	//	delete m;
	//}
	d_balls.clear();
	d_balls.shrink_to_fit();

	d_planes.clear();
	d_planes.shrink_to_fit();
	printf("GPU freed\n");



}


// creates half-space ax + by + cz < d
void Simulation::createPlane(const Vec3d& abc, const double d, const double FRICTION_K, const double FRICTION_S) { // creates half-space ax + by + cz < d
	if (ENDED) { throw std::runtime_error("The simulation has ended. New objects cannot be created."); }
	ContactPlane* new_plane = new ContactPlane(abc, d);
	assert(FRICTION_K >= 0);// make sure the friction coefficient are meaningful values
	assert(FRICTION_S >= 0);
	new_plane->_FRICTION_K = FRICTION_K;
	new_plane->_FRICTION_S = FRICTION_S;
	constraints.push_back(new_plane);

	CudaContactPlane cuda_contact_plane;
	cuda_contact_plane._normal = new_plane->_normal;
	cuda_contact_plane._offset = d;
	cuda_contact_plane._FRICTION_K = FRICTION_K;
	cuda_contact_plane._FRICTION_S = FRICTION_S;

	d_planes.push_back(cuda_contact_plane);

	//d_planes.push_back(CudaContactPlane(*new_plane));
	update_constraints = true;
}

void Simulation::createBall(const Vec3d& center, const double r) { // creates ball with radius r at position center
	if (ENDED) { throw std::runtime_error("The simulation has ended. New constraints cannot be added."); }
	Ball* new_ball = new Ball(center, r);
	constraints.push_back(new_ball);

	CudaBall cuda_ball;
	cuda_ball._center = center;
	cuda_ball._radius = r;
	d_balls.push_back(cuda_ball);

	//d_balls.push_back(CudaBall(*new_ball));
	update_constraints = true;
}

void Simulation::clearConstraints() { // clears global constraints only
	constraints.clear();
	update_constraints = true;
}

#ifdef DEBUG_ENERGY
double Simulation::energy() { // compute total energy of the system
	getAll();
	cudaDeviceSynchronize();
	double e_potential = 0; // potential energy
	double e_kinetic = 0; //kinetic energy
	for (int i = 0; i < mass.num; i++)
	{
		e_potential += -global_acc.dot(mass.pos[i]) * mass.m[i];
		e_kinetic += 0.5 * mass.vel[i].SquaredSum() * mass.m[i];
	}
	for (int i = 0; i < spring.num; i++)
	{
		e_potential += 0.5 * spring.k[i] * pow((mass.pos[spring.left[i]] - mass.pos[spring.right[i]]).norm() - spring.rest[i], 2);
	}
	return e_potential + e_kinetic;
}
#endif // DEBUG_ENERGY

#ifdef GRAPHICS
void Simulation::setViewport(const Vec3d& camera_position, const Vec3d& target_location, const Vec3d& up_vector) {
	this->camera_pos = camera_position;
	this->camera_dir = (target_location - camera_position).normalize();
	this->camera_up = up_vector;

	// initialized the camera_horizontal reference direction
	camera_href_dir = Vec3d(1, 0, 0);
	if (abs(camera_href_dir.dot(camera_up)) > 0.9) { camera_href_dir = Vec3d(0, 1, 0);}
	camera_href_dir = camera_href_dir.decompose(camera_up).normalize();

	if (STARTED) { computeMVP(); }
}
void Simulation::moveViewport(const Vec3d& displacement) {
	this->camera_pos += displacement;
	if (STARTED) { computeMVP(); } // compute perspective projection matrix
}
void Simulation::computeMVP(bool update_view) {
	// http://www.opengl-tutorial.org/beginners-tutorials/tutorial-3-matrices/#cumulating-transformations--the-modelviewprojection-matrix
	int width, height;
	glfwGetFramebufferSize(window, &width, &height); // check if window is resized
	bool is_resized = width != window_width || height != window_height;
	if (is_resized) { // window is resized
		glfwGetFramebufferSize(window, &window_width, &window_height);
		// Projection matrix : 60� Field of View, 4:3 ratio, display range : 0.01 unit <-> 100 units
		this->Projection = glm::perspective(glm::radians(60.0f), (float)window_width / (float)window_height, 0.01f, 100.0f);
	}
	if (update_view) {
		// Camera matrix
		this->View = glm::lookAt(
			glm::vec3(camera_pos.x, camera_pos.y, camera_pos.z), // camera position in World Space
			glm::vec3(camera_pos.x + camera_dir.x,
				camera_pos.y + camera_dir.y,
				camera_pos.z + camera_dir.z),	// look at position
			glm::vec3(camera_up.x, camera_up.y, camera_up.z));  // camera up vector (set to 0,-1,0 to look upside-down)
	}
	if (is_resized || update_view) {
		this->MVP = Projection * View; // Remember, matrix multiplication is the other way around
	}
}


/* create vertex buffer object // modified form cuda samples: simpleGL.cu
*/
void Simulation::createVBO(GLuint* vbo, struct cudaGraphicsResource** vbo_res, size_t size, unsigned int vbo_res_flags, GLenum buffer_type) {
	assert(vbo);
	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(buffer_type, *vbo);
	// initialize buffer object
	glBufferData(buffer_type, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(buffer_type, 0);
	// register this buffer object with CUDA
	gpuErrchk(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));
	//SDK_CHECK_ERROR_GL();
}

void Simulation::resizeVBO(GLuint* vbo, struct cudaGraphicsResource** vbo_res, size_t size, unsigned int vbo_res_flags, GLenum buffer_type) {
	//TODO
	deleteVBO(vbo, *vbo_res);
	createVBO(vbo, vbo_res, size, cudaGraphicsMapFlagsNone, buffer_type);//TODO CHANGE TO WRITE ONLY
}

/* delelte vertex buffer object//modified form cuda samples: simpleGL.cu */
void Simulation::deleteVBO(GLuint* vbo, struct cudaGraphicsResource* vbo_res)
{
	// unregister this buffer object with CUDA
	gpuErrchk(cudaGraphicsUnregisterResource(vbo_res));
	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);
	*vbo = 0;
}

inline void Simulation::generateBuffers() {
	

	createVBO(&vbo_vertex, &cuda_resource_vertex, mass.num * sizeof(float3), cudaGraphicsMapFlagsNone, GL_ARRAY_BUFFER);//TODO CHANGE TO WRITE ONLY
	createVBO(&vbo_color, &cuda_resource_color, mass.num * sizeof(float3), cudaGraphicsMapFlagsNone, GL_ARRAY_BUFFER);
	createVBO(&vbo_edge, &cuda_resource_edge, spring.num * sizeof(uint2), cudaGraphicsMapFlagsNone, GL_ELEMENT_ARRAY_BUFFER);
}

inline void Simulation::resizeBuffers() {
	////TODO>>>>>>
	resizeVBO(&vbo_vertex, &cuda_resource_vertex, mass.num * sizeof(float3), cudaGraphicsMapFlagsNone, GL_ARRAY_BUFFER);//TODO CHANGE TO WRITE ONLY
	resizeVBO(&vbo_edge, &cuda_resource_edge, spring.num * sizeof(uint2), cudaGraphicsMapFlagsNone, GL_ELEMENT_ARRAY_BUFFER);
	resizeVBO(&vbo_color, &cuda_resource_color, mass.num * sizeof(float3), cudaGraphicsMapFlagsNone, GL_ARRAY_BUFFER);

	resize_buffers = false;
}

__global__ void updateVertices(float3* __restrict__ gl_ptr, const Vec3d* __restrict__  pos, const int num_mass) {
	// https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
	// https://devblogs.nvidia.com/cuda-pro-tip-optimize-pointer-aliasing/
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_mass; i += blockDim.x * gridDim.x) {
		Vec3d pos_i = pos[i];
		gl_ptr[i].x = (float)pos_i.x;
		gl_ptr[i].y = (float)pos_i.y;
		gl_ptr[i].z = (float)pos_i.z;
	}
}

__global__ void updateIndices(uint2* __restrict__ gl_ptr, const Vec2i* __restrict__ edge, const int num_spring) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_spring; i += blockDim.x * gridDim.x) {
		Vec2i e = edge[i];

		gl_ptr[i].x = (unsigned int)e.x; // todo check if this is needed
		gl_ptr[i].y = (unsigned int)e.y;
	}
}

__global__ void updateColors(float3* __restrict__ gl_ptr, const Vec3d* __restrict__ color, const int num_mass) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_mass; i += blockDim.x * gridDim.x) {
		Vec3d color_i = color[i];
		gl_ptr[i].x = (float)color_i.x;
		gl_ptr[i].y = (float)color_i.y;
		gl_ptr[i].z = (float)color_i.z;
	}
}

void Simulation::updateBuffers() { // todo: check the kernel call
	// map OpenGL buffer object for writing from CUDA
	{ // position update
		gpuErrchk(cudaGraphicsMapResources(1, &cuda_resource_vertex, 0));
		size_t num_bytes;
		gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&dptr_vertex, &num_bytes,
			cuda_resource_vertex));
		////printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);
		updateVertices << <massBlocksPerGrid, MASS_THREADS_PER_BLOCK, 0, stream[0] >> > (dptr_vertex, d_mass.pos, mass.num);
		// unmap buffer object
		gpuErrchk(cudaGraphicsUnmapResources(1, &cuda_resource_vertex, 0));
	}
	if (update_colors) { // color update
		gpuErrchk(cudaGraphicsMapResources(1, &cuda_resource_color, 0));
		size_t num_bytes;
		gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&dptr_color, &num_bytes,
			cuda_resource_color));
		////printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);
		updateColors << <massBlocksPerGrid, MASS_THREADS_PER_BLOCK, 0, stream[1] >> > (dptr_color, d_mass.color, mass.num);
		// unmap buffer object
		gpuErrchk(cudaGraphicsUnmapResources(1, &cuda_resource_color, 0));
		update_colors = false;
	}
	if (update_indices) { // edge update
		gpuErrchk(cudaGraphicsMapResources(1, &cuda_resource_edge, 0));
		size_t num_bytes;
		gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&dptr_edge, &num_bytes,
			cuda_resource_edge));
		////printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);
		updateIndices << <springBlocksPerGrid, THREADS_PER_BLOCK, 0, stream[2] >> > (dptr_edge, d_spring.edge, spring.num);
		// unmap buffer object
		gpuErrchk(cudaGraphicsUnmapResources(1, &cuda_resource_edge, 0));
		update_indices = false;
	}
}

void Simulation::updateVertexBuffers() {

	gpuErrchk(cudaGraphicsMapResources(1, &cuda_resource_vertex, 0));
	size_t num_bytes;
	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&dptr_vertex, &num_bytes,
		cuda_resource_vertex));
	////printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);
	updateVertices << <massBlocksPerGrid, MASS_THREADS_PER_BLOCK, 0, stream[0] >> > (dptr_vertex, d_mass.pos, mass.num);
	// unmap buffer object
	gpuErrchk(cudaGraphicsUnmapResources(1, &cuda_resource_vertex, 0));
}

inline void Simulation::draw() {
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, this->vbo_vertex);
	glPointSize(this->point_size);
	glLineWidth(this->line_width);
	glVertexAttribPointer(
		0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0            // array buffer offset
	);

	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, this->vbo_color);
	glVertexAttribPointer(
		1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
		3,                                // size
		GL_FLOAT,                         // type
		GL_FALSE,                         // normalized?
		0,                                // stride
		(void*)0                          // array buffer offset
	);

	glDrawArrays(GL_POINTS, 0, mass.num); // 3 indices starting at 0 -> 1 triangle

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_edge);

	glDrawElements(GL_LINES, 2 * spring.num, GL_UNSIGNED_INT, (void*)0); // 2 indices for a line

	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(0);
}


void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}
void Simulation::createGLFWWindow() {
	// Initialise GLFW
	if (!glfwInit()) { throw(std::runtime_error("Failed to initialize GLFW\n")); }
	////// MSAA: multisampling
	glfwWindowHint(GLFW_SAMPLES, 0); // #samples to use for multisampling. Zero disables multisampling.
	glEnable(GL_MULTISAMPLE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); // use GLSL 4.6
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // meke opengl forward compatible
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); //We don't want the old OpenGL
	glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
	glfwSwapInterval(0);// disable vsync
	//glfwSwapInterval(1);// enable vsync

	//// Open a window and create its OpenGL context
	//auto monitor = glfwGetPrimaryMonitor();
	//const GLFWvidmode* mode = glfwGetVideoMode(monitor);
	//GLFWwindow* window = glfwCreateWindow(mode->width, mode->height, "CUDA Physics Simulation", NULL, NULL);
	window = glfwCreateWindow(1920, 1080, "CUDA Physics Simulation", NULL, NULL);

	if (window == NULL) {
		fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible.\n");
		getchar();
		glfwTerminate();
		exit(1);
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	glEnable(GL_CULL_FACE); // FACE CULLING :https://learnopengl.com/Advanced-OpenGL/Face-culling
	glCullFace(GL_FRONT);
	glFrontFace(GL_CCW);


	glEnable(GL_DEPTH_TEST);
	//    // Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);
	// Initialize GLEW
	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		getchar();
		glfwTerminate();
		exit(1);
	}
	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	// reset window color
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
}

#endif
