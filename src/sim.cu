/*modified from the orginal Titan simulation libaray:https://github.com/jacobaustin123/Titan
ref: J. Austin, R. Corrales-Fatou, S. Wyetzner, and H. Lipson, �Titan: A Parallel Asynchronous Library for Multi-Agent and Soft-Body Robotics using NVIDIA CUDA,� ICRA 2020, May 2020.
*/

#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#endif // !CUDA_API_PER_THREAD_DEFAULT_STREAM

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
constexpr int THREADS_PER_BLOCK = 128;
constexpr int MASS_THREADS_PER_BLOCK = 128;


GLenum glCheckError_(const char* file, int line)
{
	GLenum errorCode;
	while ((errorCode = glGetError()) != GL_NO_ERROR)
	{
		std::string error;
		switch (errorCode)
		{
		case GL_INVALID_ENUM:                  error = "INVALID_ENUM"; break;
		case GL_INVALID_VALUE:                 error = "INVALID_VALUE"; break;
		case GL_INVALID_OPERATION:             error = "INVALID_OPERATION"; break;
		case GL_STACK_OVERFLOW:                error = "STACK_OVERFLOW"; break;
		case GL_STACK_UNDERFLOW:               error = "STACK_UNDERFLOW"; break;
		case GL_OUT_OF_MEMORY:                 error = "OUT_OF_MEMORY"; break;
		case GL_INVALID_FRAMEBUFFER_OPERATION: error = "INVALID_FRAMEBUFFER_OPERATION"; break;
		}
		std::cout << error << " | " << file << " (" << line << ")" << std::endl;
	}
	return errorCode;
}
#define glCheckError() glCheckError_(__FILE__, __LINE__) 

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


//__global__ void controlUpdate(Vec3d* __restrict__ mass_pos, const JOINT joint) {
//	int i = blockIdx.x * blockDim.x + threadIdx.x;
//	if (i < joint.anchors.num) {
//		Vec2i anchor_edge = joint.anchors.edge[i];
//		Vec3d rotation_axis = (mass_pos[anchor_edge.y] - mass_pos[anchor_edge.x]).normalize();
//		Vec3d x_left = mass_pos[joint.anchors.leftCoord[i] + 1] - mass_pos[joint.anchors.leftCoord[i]];//oxyz
//		Vec3d x_right = mass_pos[joint.anchors.rightCoord[i] + 1] - mass_pos[joint.anchors.rightCoord[i]];//oxyz
//		double angle = signedAngleBetween(x_left, x_right, rotation_axis); //joint angle in [-pi,pi]
//
//		double delta_angle = angle - joint_pos[i];
//		if (delta_angle > M_PI) {
//			joint_vel[i] = (delta_angle - 2 * M_PI) / (NUM_QUEUED_KERNELS * dt);
//		}
//		else if (delta_angle > -M_PI) {
//			joint_vel[i] = delta_angle / (NUM_QUEUED_KERNELS * dt);
//		}
//		else {
//			joint_vel[i] = (delta_angle + 2 * M_PI) / (NUM_QUEUED_KERNELS * dt);
//		}
//		joint_pos[i] = angle;
//		}
//}

#endif // ROTATION


//Simulation::Simulation() {
//	//cudaSetDevice(1);
//	for (int i = 0; i < NUM_CUDA_STREAM; ++i) { // lower i = higher priority
//		cudaStreamCreateWithPriority(&stream[i], cudaStreamDefault, i);// create extra cuda stream: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution
//	}
//}

Simulation::Simulation(size_t num_mass, size_t num_spring, size_t num_joint, size_t num_triangle):
	mass(num_mass, true), // allocate host
	d_mass(num_mass, false),// allocate device
	spring(num_spring, true),// allocate host
	d_spring(num_spring, false),// allocate device
	triangle(num_triangle,true),//allocate host
	d_triangle(num_triangle, false),//allocate device
	joint_control(num_joint,true) // joint controller, must also call reset, see update_physics()
#ifdef UDP
	,udp_server(port_local, port_remote, ip_remote, num_joint)// port_local,port_remote,ip_remote,num_joint
#endif //UDP
{
	//cudaDeviceSynchronize();
	for (int i = 0; i < NUM_CUDA_STREAM; ++i) { // lower i = higher priority
		cudaStreamCreateWithPriority(&stream[i], cudaStreamDefault, i);// create extra cuda stream: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution
	}

}

void Simulation::getAll() {//copy from gpu
	mass.copyFrom(d_mass, stream[CUDA_MEMORY_STREAM]); // mass
	spring.copyFrom(d_spring, stream[CUDA_MEMORY_STREAM]);// spring
#ifdef GRAPHICS
	triangle.copyFrom(d_triangle, stream[CUDA_MEMORY_STREAM]);// triangle
#endif //GRAPHICS
	//cudaDeviceSynchronize();
}

void Simulation::setAll() {//copy form cpu
	d_mass.copyFrom(mass, stream[CUDA_MEMORY_STREAM]);
	d_spring.copyFrom(spring, stream[CUDA_MEMORY_STREAM]);
#ifdef GRAPHICS
	d_triangle.copyFrom(triangle, stream[CUDA_MEMORY_STREAM]);
#endif //GRAPHICS
	//cudaDeviceSynchronize();
}

void Simulation::setMass() {
	d_mass.copyFrom(mass, stream[NUM_CUDA_STREAM - 1]);
}

inline int Simulation::computeBlocksPerGrid(const int threadsPerBlock, const int num) {
	int blocksPerGrid = (num - 1 + threadsPerBlock) / threadsPerBlock;
	assert(blocksPerGrid <= MAX_BLOCKS);//TODO: kernel has a hard limit on MAX_BLOCKS
	return blocksPerGrid;
}

inline void Simulation::updateCudaParameters() {
	massBlocksPerGrid = computeBlocksPerGrid(MASS_THREADS_PER_BLOCK, mass.size());
	springBlocksPerGrid = computeBlocksPerGrid(THREADS_PER_BLOCK, spring.size());
	jointBlocksPerGrid = computeBlocksPerGrid(MASS_THREADS_PER_BLOCK, joint.points.size());
#ifdef GRAPHICS
	triangleBlocksPerGrid = computeBlocksPerGrid(THREADS_PER_BLOCK, triangle.size());
#endif //GRAPHICS

}

void Simulation::setBreakpoint(const double time) {
	if (ENDED) { throw std::runtime_error("Simulation has ended. Can't modify simulation after simulation end."); }
	bpts.insert(time); // TODO mutex breakpoints
}

/*pause the simulation at (simulation) time t [s] */
void Simulation::pause(const double t) {
	if (ENDED) { throw std::runtime_error("Simulation has ended. can't call control functions."); }
	setBreakpoint(t);

	//waitForEvent();

	//// Wait until main() sends data
	std::unique_lock<std::mutex> lck(mutex_running);
	SHOULD_RUN = false;
	cv_running.notify_all();
	cv_running.wait(lck, [this] {return !RUNNING; });
}


void Simulation::resume() {
	if (ENDED) { throw std::runtime_error("Simulation has ended. Cannot resume simulation."); }
	if (!STARTED) { throw std::runtime_error("Simulation has not started. Cannot resume before calling sim.start()."); }
	if (mass.num == 0) { throw std::runtime_error("No masses have been added. Add masses before simulation starts."); }
	updateCudaParameters();
	cudaDeviceSynchronize();
	std::unique_lock<std::mutex> lck(mutex_running);
	SHOULD_RUN = true;
	cv_running.notify_all();
}

void Simulation::waitForEvent() {
	if (ENDED) { throw std::runtime_error("Simulation has ended. can't call control functions."); }
	//while (RUNNING) {
	//	std::this_thread::sleep_for(std::chrono::nanoseconds(100));
	//}
	std::unique_lock<std::mutex> lck(mutex_running);
	cv_running.wait(lck, [this] {return !RUNNING; });
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
	//memset(joint_vel_cmd, 0, nbytes);
	//memset(joint_vel, 0, nbytes);
	//memset(joint_pos, 0, nbytes);
	//memset(joint_vel_desired, 0, nbytes);
	//memset(joint_vel_error, 0, nbytes);
	//memset(joint_pos_error, 0, nbytes);

	//for (int i = 0; i < joint.size(); i++) {
	//	joint_vel_cmd[i] = 0.;
	//	joint_vel[i] = 0.;
	//	joint_pos[i] = 0.;
	//	joint_vel_desired[i] = 0.;
	//	joint_vel_error[i] = 0.;
	//	joint_pos_error[i] = 0.;
	//}
	joint_control.reset(backup_mass, backup_joint);
	body.init(backup_mass, id_oxyz_start); // init body frame

}

//void Simulation::setMaxJointSpeed(double max_joint_vel) {
//	this->max_joint_vel = max_joint_vel;
//	max_joint_vel_error = max_joint_vel / k_vel;
//	max_joint_pos_error = max_joint_vel / k_pos;
//}


void Simulation::start() {
	if (ENDED) { throw std::runtime_error("The simulation has ended. Cannot call sim.start() after the end of the simulation."); }
	if (mass.num == 0) { throw std::runtime_error("No masses have been added. Please add masses before starting the simulation."); }
	printf("Starting simulation with %d masses and %d springs\n", mass.num, spring.num);
	RUNNING = true;
	SHOULD_RUN = true;
	STARTED = true;

	T = 0;

	
	if (dt == 0.0) { // if dt hasn't been set by the user.
		dt = 0.01; // min delta
	}
	updateCudaParameters();

	d_constraints.d_balls = thrust::raw_pointer_cast(&d_balls[0]);
	d_constraints.d_planes = thrust::raw_pointer_cast(&d_planes[0]);
	d_constraints.num_balls = d_balls.size();
	d_constraints.num_planes = d_planes.size();

	SHOULD_UPDATE_CONSTRAINT = false;

	//cudaMallocHost((void**)&joint_pos_error, joint.size() * sizeof(double));//initialize joint speed error integral array 
	//cudaMallocHost((void**)&joint_vel_error, joint.size() * sizeof(double));//initialize joint speed error array 
	//cudaMallocHost((void**)&joint_vel_cmd, joint.size() * sizeof(double));//initialize joint speed (commended) array 
	//cudaMallocHost((void**)&joint_vel_desired, joint.size() * sizeof(double));//initialize joint speed (desired) array 
	//cudaMallocHost((void**)&joint_vel, joint.size() * sizeof(double));//initialize joint speed (measured) array 
	//cudaMallocHost((void**)&joint_pos, joint.size() * sizeof(double));//initialize joint angle (measured) array 

	setAll();// copy mass and spring to gpu

	backupState();// backup the robot mass/spring/joint state

#ifdef UDP
	udp_server.run();
#endif //UDP

	thread_physics_update = std::thread(&Simulation::update_physics, this); //TODO: thread
#ifdef GRAPHICS
	thread_graphics_update = std::thread(&Simulation::update_graphics, this); //TODO: thread
#endif// Graphics

	{
		int device;
		cudaGetDevice(&device);
		printf("cuda device: %d\n", device);
	}

}

void Simulation::update_physics() { // repeatedly start next

	//cudaDeviceProp deviceProp;
	//cudaGetDeviceProperties(&deviceProp, 0);
	//if (deviceProp.cooperativeLaunch == 0) {
	//	printf("not supported");
	//	exit(-1);
	//}

	cudaDeviceSynchronize();//sync before while loop
	auto start = std::chrono::steady_clock::now();
	int k_udp = 0;
	int k_dynamics = 0;

#ifdef DEBUG_ENERGY
	energy_start = energy(); // compute the total energy of the system at T=0
#endif // DEBUG_ENERGY

	joint_control.reset(mass, joint);// reset joint controller, must do
	body.init(mass, id_oxyz_start); // init body frame

	while (true) {
		if (!bpts.empty() && *bpts.begin() <= T) {// paused when a break p
			cudaDeviceSynchronize(); // synchronize before updating the springs and mass positions
		//            std::cout << "Breakpoint set for time " << *bpts.begin() << " reached at simulation time " << T << "!" << std::endl;
			bpts.erase(bpts.begin());
			if (bpts.empty()) { SHOULD_END = true; }
			if (SHOULD_END) {
				auto end = std::chrono::steady_clock::now();
				double duration = (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.;//[seconds]
				double sim_time_ratio = T / duration;
				double spring_update_rate = ((double)spring.num) / dt * sim_time_ratio;
				printf("Elapsed time:%.2f s for %.2f simulation time (%.2f); # %.2e spring update/s\n",
					duration, T, sim_time_ratio, spring_update_rate);

				//for (Constraint* c : constraints) {
				//	delete c;
				//}

				std::unique_lock<std::mutex> lck(mutex_running); // refer to:https://en.cppreference.com/w/cpp/thread/condition_variable
#ifdef GRAPHICS
				GRAPHICS_SHOULD_END = true;
				cv_running.notify_all(); //notify others RUNNING = false
				cv_running.wait(lck, [this] {return GRAPHICS_ENDED; });
#endif
				//#ifdef UDP
				//				udp_server.close();
				//#endif
				GPU_DONE = true;
				RUNNING = false;
				printf("GPU done\n");
				return;
			}
			{	// condition variable 
				std::unique_lock<std::mutex> lck(mutex_running); // refer to:https://en.cppreference.com/w/cpp/thread/condition_variable
				RUNNING = false;
				cv_running.notify_all(); //notify others RUNNING = false
				cv_running.wait(lck, [this] {return SHOULD_RUN; }); // wait unitl SHOULD_RUN is signaled
				RUNNING = true;
				//lck.unlock();// Manual unlocking before notifying, to avoid waking up the waiting thread only to block again
				cv_running.notify_all(); // notifiy others RUNNING = true;
			}

			//			// TODO NOTIFY THE OTHER THREAD
			//			if (SHOULD_UPDATE_CONSTRAINT) {
			//				d_constraints.d_balls = thrust::raw_pointer_cast(&d_balls[0]);
			//				d_constraints.d_planes = thrust::raw_pointer_cast(&d_planes[0]);
			//				d_constraints.num_balls = d_balls.size();
			//				d_constraints.num_planes = d_planes.size();
			//#ifdef GRAPHICS
			//				for (Constraint* c : constraints) { // generate buffers for constraint objects
			//					if (!c->_initialized)
			//						c->generateBuffers();
			//				}
			//				SHOULD_UPDATE_CONSTRAINT = false;
			//#endif // GRAPHICS
			//			}

			continue;
		}

		////cudaDeviceSynchronize();
		//cudaEvent_t event; // an event that tel
		//cudaEventCreateWithFlags(&event, cudaEventDisableTiming); // create event,disable timing for faster speed:https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf
		//cudaEvent_t event_rotation;
		//cudaEventCreateWithFlags(&event_rotation, cudaEventDisableTiming);
		for (int i = 0; i < NUM_QUEUED_KERNELS; i++) {
			if (k_dynamics % NUM_UPDATE_PER_ROTATION==0) {
#ifdef ROTATION
				rotateJoint << <jointBlocksPerGrid, MASS_THREADS_PER_BLOCK, 0, stream[CUDA_DYNAMICS_STREAM] >> > (d_mass.pos, d_joint);
				SpringUpateReset << <springBlocksPerGrid, THREADS_PER_BLOCK, 0, stream[CUDA_DYNAMICS_STREAM] >> > (d_mass, d_spring);
				MassUpate << <massBlocksPerGrid, MASS_THREADS_PER_BLOCK, 0, stream[CUDA_DYNAMICS_STREAM] >> > (d_mass, d_constraints, global_acc, dt);

				//SpringUpate << <springBlocksPerGrid, THREADS_PER_BLOCK >> > (d_mass, d_spring);
				//massUpdateAndRotate << <massBlocksPerGrid + jointBlocksPerGrid, MASS_THREADS_PER_BLOCK >> > (d_mass, d_constraints, d_joint, global_acc, dt);
				//SpringUpateReset << <springBlocksPerGrid, THREADS_PER_BLOCK >> > (d_mass, d_spring);
				//MassUpate << <massBlocksPerGrid, MASS_THREADS_PER_BLOCK >> > (d_mass, d_constraints, global_acc, dt);

				//gpuErrchk(cudaPeekAtLastError());
			}
			else {
				SpringUpate << <springBlocksPerGrid, THREADS_PER_BLOCK, 0, stream[CUDA_DYNAMICS_STREAM] >> > (d_mass, d_spring);
				MassUpate << <massBlocksPerGrid, MASS_THREADS_PER_BLOCK, 0, stream[CUDA_DYNAMICS_STREAM] >> > (d_mass, d_constraints, global_acc, dt);
				//gpuErrchk(cudaPeekAtLastError());
//			//cudaEventRecord(event, 0);
//			//cudaStreamWaitEvent(stream[0], event, 0);
//			//cudaEventRecord(event_rotation, stream[0]);
//			//cudaStreamWaitEvent(NULL, event_rotation, 0);
			}
#else
				SpringUpate << <springBlocksPerGrid, THREADS_PER_BLOCK, 0, stream[CUDA_DYNAMICS_STREAM] >> > (d_mass, d_spring);
				MassUpate << <massBlocksPerGrid, MASS_THREADS_PER_BLOCK, 0, stream[CUDA_DYNAMICS_STREAM] >> > (d_mass, d_constraints, global_acc, dt);
#endif // ROTATION
			k_dynamics++;
		}


		//cudaEventDestroy(event);//destroy the event, necessary to prevent memory leak
		//cudaEventDestroy(event_rotation);//destroy the event, necessary to prevent memory leak

		T += NUM_QUEUED_KERNELS * dt;

		//if (fmod(T, 1. / 100.0) < NUM_QUEUED_KERNELS * dt) {
		mass.CopyPosVelAccFrom(d_mass, stream[CUDA_MEMORY_STREAM]);
		cudaDeviceSynchronize();

		//Vec3d com_pos = mass.pos[id_oxyz_start];//body center of mass position
		//Vec3d com_vel = mass.vel[id_oxyz_start];
		//Vec3d com_acc = mass.acc[id_oxyz_start];//body center of mass acceleration
		////Vec3d com_acc = mass.acc[id_oxyz_start];
		////for (size_t i = id_oxyz_start+1; i < id_oxyz_start + 7; i++)
		////{
		////	com_acc += mass.acc[i];//average over o,x,y,z,-x,-y,-z
		////}
		////com_acc /= 7;

		//std::chrono::steady_clock::time_point ct_begin = std::chrono::steady_clock::now();
		////std::chrono::steady_clock::time_point ct_end = std::chrono::steady_clock::now();
		////std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(ct_end - ct_begin).count() << "[micro s]" << std::endl;
		////std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (ct_end - ct_begin).count() << "[ns]" << std::endl;

		joint_control.update(mass, joint, NUM_QUEUED_KERNELS * dt);
		// update joint speed
		for (int i = 0; i < joint.anchors.num; i++) { // compute joint angles and angular velocity
			joint.anchors.theta[i] = NUM_UPDATE_PER_ROTATION * joint_control.joint_vel_cmd[i] * dt;// update joint speed
		}
		d_joint.anchors.copyThetaFrom(joint.anchors, stream[CUDA_MEMORY_STREAM]);

#ifdef DEBUG_ENERGY
		if (fmod(T, 1. / 10.0) < NUM_QUEUED_KERNELS * dt) {
			double e = energy();
			//if (abs(e - energy_start) > energy_deviation_max) {
			//	energy_deviation_max = abs(e - energy_start);
			//	printf("%.3f\t%.3f\n", T, energy_deviation_max / energy_start);
			//}
			//else { printf("%.3f\r", T); }
			printf("%.3f\t%.3f\n", T, e);
		}

#endif // DEBUG_ENERGY

		if (k_udp % NUM_UDP_MULTIPLIER == 0) {

			body.update(mass, id_oxyz_start, NUM_UDP_MULTIPLIER*NUM_QUEUED_KERNELS* dt);

#ifdef UDP
			//msg_send.T_prev = msg_send.T;//update previous time
			udp_server.msg_send.T = T;//update time

			for (auto i = 0; i < joint.size(); i++)
			{
				udp_server.msg_send.joint_pos[i] = joint_control.joint_pos[i];
				udp_server.msg_send.joint_vel[i] = joint_control.joint_vel[i];
				udp_server.msg_send.actuation[i] = joint_control.joint_vel_cmd[i] / joint_control.max_joint_vel[i];
				//msg_send.joint_vel_desired[i] = joint_control.joint_vel_desired[i];//desired joint velocity at last command
			}

			udp_server.msg_send.com_acc = body.acc;
			udp_server.msg_send.com_vel = body.vel;
			udp_server.msg_send.com_pos = body.pos;

			udp_server.msg_send.orientation[0] = body.rot.m00;
			udp_server.msg_send.orientation[1] = body.rot.m10;
			udp_server.msg_send.orientation[2] = body.rot.m20;
			udp_server.msg_send.orientation[3] = body.rot.m01;
			udp_server.msg_send.orientation[4] = body.rot.m11;
			udp_server.msg_send.orientation[5] = body.rot.m21;

			udp_server.msg_send.ang_vel = body.ang_vel;

			udp_server.flag_should_send = true;
			// receiving message
			if (udp_server.flag_new_received) {
				udp_server.flag_new_received = false;

				switch (udp_server.msg_rec.header)
				{
				case UDP_HEADER::TERMINATE://close the program
					bpts.insert(T);
					SHOULD_END = true;
					break;
				case UDP_HEADER::RESET: // reset
					// set the reset flag to true, resetState() will be called 
					// to restore the robot mass/spring/joint state to the backedup state
					RESET = true;
					break;
				case UDP_HEADER::MOTOR_SPEED_COMMEND:
					for (int i = 0; i < joint.anchors.num; i++) {//update joint speed from received udp packet
						joint_control.joint_vel_desired[i] = udp_server.msg_rec.joint_vel_desired[i];
					}
					break;
				default:
					break;
				}
				if (fmod(T, 1. / 10.0) < NUM_QUEUED_KERNELS * dt) {// print only once in a while
					printf("%3.3f \t", T); // alternative lagged: udp_server.msg_rec.T
					for (int i = 0; i < joint.size(); i++)
					{
						printf("%3.3f ", joint_control.joint_vel_desired[i]);
					}
					printf("\r\r"); // TODO: improve speed, maybe create string and print at once
				}
			}



			//if (fmod(T, 1. / 10.0) < NUM_QUEUED_KERNELS * dt) {
			//	//printf("% 6.1f",T); // time
			//	//printf("|t");
			//	//for (int i = 0; i < joint.anchors.num; i++) {
			//	//	printf("% 4.0f", joint_pos[i] * M_1_PI * 180.0); // display joint angle in deg
			//	//}
			//	//printf("|w");
			//	//for (int i = 0; i < joint.anchors.num; i++) {
			//	//	printf("% 4.0f", joint_vel[i] * M_1_PI * 30.0); // display joint speed in RPM
			//	//}

			//	//printf("|wc");
			//	//for (int i = 0; i < joint.anchors.num; i++) {
			//	//	printf("% 4.0f", joint_vel_cmd[i] * M_1_PI * 30.0); // display joint speed in RPM
			//	//}

			//	////printf("|xy%+ 2.2f %+ 2.2f %+ 2.2f ", ox.x, ox.y, ox.z);
			//	////printf("%+ 2.2f %+ 2.2f %+ 2.2f", oy.x, oy.y, oy.z);

			//	//printf("|x%+ 6.2f %+ 6.2f %+ 6.2f", com_pos.x, com_pos.y, com_pos.z);
			//	//printf("|a%+ 6.1f %+ 6.1f %+ 6.1f", com_acc.x, com_acc.y, com_acc.z);
			//	//printf("\r\r");
			//}
			////printf("\n");
		//}
#endif // UDP
	}
		k_udp += 1;

		if (RESET) {
			cudaDeviceSynchronize();
			RESET = false;
			resetState();// restore the robot mass/spring/joint state to the backedup state
			cudaDeviceSynchronize();
		}
	}
}

#ifdef GRAPHICS
void Simulation::update_graphics() {



	createGLFWWindow(); // create a window with  width and height

	int glDeviceId;// todo:this output wrong number, maybe it is a cuda bug...
	unsigned int glDeviceCount;
	cudaGLGetDevices(&glDeviceCount, &glDeviceId, 1u, cudaGLDeviceListAll);
	printf("openGL device: %u\n", glDeviceId);

	glGenVertexArrays(1, &VertexArrayID);//GLuint VertexArrayID;
	glBindVertexArray(VertexArrayID);
	// Create and compile our GLSL program from the shaders
	programID = LoadShaders("shaderVertex.glsl", "shaderFragment.glsl"); //
	glUseProgram(programID);// Use our shader
	// Get a handle for our "MVP" uniform
	computeMVP(); // compute perspective projection matrix
	MatrixID = glGetUniformLocation(programID, "MVP"); // doesn't seem to be necessary
	generateBuffers(); // generate buffers for all masses and springs

	for (Constraint* c : constraints) { // generate buffers for constraint objects
		c->generateBuffers();
	}
	//updateBuffers();//Todo might not need?
	cudaDeviceSynchronize();//sync before while loop

	//double T_previous_update = T;	

	// check for errors
	GLenum error = glGetError();
	if (error != GL_NO_ERROR)
		std::cerr << "OpenGL Error " << error << std::endl;

	while (!GRAPHICS_SHOULD_END) {

		std::this_thread::sleep_for(std::chrono::microseconds(int(1e6 / 60.02)));// TODO fix race condition

		//if (T- T_previous_update>1.0/60.1) 
		{ // graphics update loop
			if (resize_buffers) {
				resizeBuffers(); // needs to be run from GPU thread
				updateBuffers(); // full update
			}
			else {
				updateVertexBuffers(); // partial update
			}
			//cudaDeviceSynchronize(); // synchronize before updating the springs and mass positions
			//T_previous_update = T;

			Vec3d com_pos = mass.pos[id_oxyz_start];// center of mass position (anchored body center)

			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear screen

			double t_lerp = 0.01;
			double speed_multiplier = 0.02;

			if (glfwGetKey(window, GLFW_KEY_W)) { camera_h_offset -= 0.05; }//camera moves closer
			else if (glfwGetKey(window, GLFW_KEY_S)) {camera_h_offset += 0.05;}//camera moves away
			if (glfwGetKey(window, GLFW_KEY_A)) {camera_yaw -= 0.05;} //camera moves left
			else if (glfwGetKey(window, GLFW_KEY_D)) {camera_yaw += 0.05;}//camera moves right
			if (glfwGetKey(window, GLFW_KEY_Q)) {camera_up_offset -= 0.05;} // camera moves down
			else if (glfwGetKey(window, GLFW_KEY_E)) {camera_up_offset += 0.05;}// camera moves up

			if (glfwGetKey(window, GLFW_KEY_UP)) {
				if (joint_control.joint_vel_desired[0] < joint_control.max_joint_vel[0]) { joint_control.joint_vel_desired[0] += speed_multiplier; }
				if (joint_control.joint_vel_desired[1] < joint_control.max_joint_vel[1]) { joint_control.joint_vel_desired[1] += speed_multiplier; }
				if (joint_control.joint_vel_desired[2] > -joint_control.max_joint_vel[2]) { joint_control.joint_vel_desired[2] -= speed_multiplier; }
				if (joint_control.joint_vel_desired[3] > -joint_control.max_joint_vel[3]) { joint_control.joint_vel_desired[3] -= speed_multiplier; }
			}
			else if (glfwGetKey(window, GLFW_KEY_DOWN)) {
				if (joint_control.joint_vel_desired[0] > -joint_control.max_joint_vel[0]) { joint_control.joint_vel_desired[0] -= speed_multiplier; }
				if (joint_control.joint_vel_desired[1] > -joint_control.max_joint_vel[1]) { joint_control.joint_vel_desired[1] -= speed_multiplier; }
				if (joint_control.joint_vel_desired[2] < joint_control.max_joint_vel[2]) { joint_control.joint_vel_desired[2] += speed_multiplier; }
				if (joint_control.joint_vel_desired[3] < joint_control.max_joint_vel[3]) { joint_control.joint_vel_desired[3] += speed_multiplier; }
			}
			if (glfwGetKey(window, GLFW_KEY_LEFT)) {
				for (int i = 0; i < joint.size(); i++)
				{
					if (joint_control.joint_vel_desired[i] > -joint_control.max_joint_vel[i]) { joint_control.joint_vel_desired[i] -= speed_multiplier; }
				}
			}
			else if (glfwGetKey(window, GLFW_KEY_RIGHT)) {
				for (int i = 0; i < joint.size(); i++)
				{
					if (joint_control.joint_vel_desired[i] < joint_control.max_joint_vel[i]) { joint_control.joint_vel_desired[i] += speed_multiplier; }
				}
			}
			else if (glfwGetKey(window, GLFW_KEY_0)) { // zero speed
				for (int i = 0; i < joint.size(); i++) { joint_control.joint_vel_desired[i] = 0.; }
			}
			else if (glfwGetKey(window, GLFW_KEY_R)) { // reset
				RESET = true;
			}

			// https://en.wikipedia.org/wiki/Slerp
			//mass.pos[id_oxyz_start].print();
			

			Vec3d camera_rotation_anchor = com_pos + (camera_up_offset - com_pos.dot(camera_up)) * camera_up;

			Vec3d camera_pos_desired = AxisAngleRotaion(camera_up, camera_href_dir * camera_h_offset, camera_yaw, Vec3d()) + camera_rotation_anchor;

			camera_pos = lerp(camera_pos, camera_pos_desired, 2*t_lerp);

			Vec3d camera_dir_desired = (com_pos - camera_pos).normalize();

			// spherical linear interpolation from camera_dir to camera_dir_new by factor t_lerp
			camera_dir = slerp(camera_dir, camera_dir_desired, t_lerp).normalize();

			computeMVP(true); // update MVP, also update camera matrix //todo

			glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);// update transformation "MVP" uniform


			draw();

			for (Constraint* c : constraints) {
				c->draw();
			}

			// Swap buffers, render screen
			glfwPollEvents();
			glfwSwapBuffers(window);

			//// check for errors
			//GLenum error = glGetError();
			//if (error != GL_NO_ERROR)
			//	std::cerr << "OpenGL Error " << error << std::endl;

			if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS || glfwWindowShouldClose(window) != 0) {
				bpts.insert(T);// break at current time T
				//exit(0); // TODO maybe deal with memory leak here. //key press exit,
				SHOULD_END = true;
			}
		}

	}

	// end the graphics
	deleteBuffers(); // delete the buffer objects
	glDeleteProgram(programID);
	glDeleteVertexArrays(1, &VertexArrayID);
	glfwTerminate(); // Close OpenGL window and terminate GLFW
	printf("\nwindow closed\n");

	{	// notify the physics thread
		std::unique_lock<std::mutex> lck(mutex_running); // could just use lock_guard
		GRAPHICS_ENDED = true;
		lck.unlock();
		cv_running.notify_all();
	}

	
	return;


}
#endif

void Simulation::execute() {


}




Simulation::~Simulation() {
	std::cout << "Simulation destructor called." << std::endl;

	if (STARTED) {
		{		
			//std::unique_lock<std::mutex> lck(mutex_running);
			//cv_running.wait(lck, [this] {return !RUNNING; }); 
			while (RUNNING) {
				std::this_thread::sleep_for(std::chrono::milliseconds(10));// TODO fix race condition
			}

		}

		ENDED = true; // TODO maybe race condition

		while (!GPU_DONE) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));// TODO fix race condition
		}
		if (thread_physics_update.joinable()) {

			thread_physics_update.join();
			printf("thread_physics_update joined\n");
		}
#ifdef GRAPHICS
		if (thread_graphics_update.joinable()) {
			thread_graphics_update.join();
			printf("thread_graphics_update joined\n");
		}
#endif
		else {
			printf("could not join GPU thread.\n");
			exit(1);
		}
#ifdef UDP
		udp_server.close();
#endif // UDP

	}
	freeGPU();
}


void Simulation::freeGPU() {
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
	SHOULD_UPDATE_CONSTRAINT = true;
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
	SHOULD_UPDATE_CONSTRAINT = true;
}

void Simulation::clearConstraints() { // clears global constraints only
	constraints.clear();
	SHOULD_UPDATE_CONSTRAINT = true;
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
	if (abs(camera_href_dir.dot(camera_up)) > 0.9) { camera_href_dir = Vec3d(0, 1, 0); }
	camera_href_dir = camera_href_dir.decompose(camera_up).normalize();

	if (STARTED) { computeMVP(); }
}
void Simulation::moveViewport(const Vec3d& displacement) {
	this->camera_pos += displacement;
	if (STARTED) { computeMVP(); } // compute perspective projection matrix
}
void Simulation::computeMVP(bool update_view) {
	// http://www.opengl-tutorial.org/beginners-tutorials/tutorial-3-matrices/#cumulating-transformations--the-modelviewprojection-matrix
	
	
	int iconified = glfwGetWindowAttrib(window, GLFW_ICONIFIED);// whether window is iconified
	int width, height;
	glfwGetFramebufferSize(window, &width, &height); // check if window is resized
	bool is_resized = !iconified && (width != framebuffer_width || (height != framebuffer_height));
	if (is_resized) {// window is resized but not iconified
		framebuffer_width = width;
		framebuffer_height = height;
		// Projection matrix : 70 deg Field of View, width:height ratio, display range : 0.01 unit <-> 100 units
		this->Projection = glm::perspective(glm::radians(70.0f), (float)framebuffer_width / (float)framebuffer_height, 0.01f, 100.0f);
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
	glCheckError(); // check opengl error code
}

/* resize the buffer object*/
void Simulation::resizeVBO(GLuint* vbo, size_t size, GLenum buffer_type) {
	glBindBuffer(buffer_type, *vbo);
	glBufferData(buffer_type, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(buffer_type, 0);
	glCheckError();// check opengl error code
}

/* delelte vertex buffer object//modified form cuda samples: simpleGL.cu */
void Simulation::deleteVBO(GLuint* vbo, struct cudaGraphicsResource* vbo_res, GLenum buffer_type)
{
	gpuErrchk(cudaGraphicsUnregisterResource(vbo_res));// unregister buffer object with CUDA
	glBindBuffer(buffer_type, *vbo);
	glDeleteBuffers(1, vbo);
	*vbo = 0;
	glCheckError();// check opengl error code
}

inline void Simulation::generateBuffers() {
	createVBO(&vbo_vertex, &cuda_resource_vertex, mass.size() * sizeof(float3), cudaGraphicsMapFlagsNone, GL_ARRAY_BUFFER);//TODO CHANGE TO WRITE ONLY
	createVBO(&vbo_color, &cuda_resource_color, mass.size() * sizeof(float3), cudaGraphicsMapFlagsNone, GL_ARRAY_BUFFER);
	createVBO(&vbo_edge, &cuda_resource_edge, spring.size() * sizeof(uint2), cudaGraphicsMapFlagsNone, GL_ELEMENT_ARRAY_BUFFER);
	createVBO(&vbo_triangle, &cuda_resource_triangle, triangle.size() * sizeof(uint3), cudaGraphicsMapFlagsNone, GL_ELEMENT_ARRAY_BUFFER);
}

inline void Simulation::resizeBuffers() {
	////TODO>>>>>>
	resizeVBO(&vbo_vertex, mass.size() * sizeof(float3), GL_ARRAY_BUFFER);//TODO CHANGE TO WRITE ONLY
	resizeVBO(&vbo_color, mass.size() * sizeof(float3), GL_ARRAY_BUFFER);
	resizeVBO(&vbo_edge, spring.size() * sizeof(uint2), GL_ELEMENT_ARRAY_BUFFER);
	resizeVBO(&vbo_triangle, triangle.size() * sizeof(uint3), GL_ELEMENT_ARRAY_BUFFER);
	resize_buffers = false;
}
inline void Simulation::deleteBuffers() {
	deleteVBO(&vbo_vertex, cuda_resource_vertex, GL_ARRAY_BUFFER);
	deleteVBO(&vbo_color, cuda_resource_color, GL_ARRAY_BUFFER);
	deleteVBO(&vbo_edge, cuda_resource_edge, GL_ELEMENT_ARRAY_BUFFER);
	deleteVBO(&vbo_triangle, cuda_resource_triangle, GL_ELEMENT_ARRAY_BUFFER);
}


// update vertex positions
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

// update line indices
__global__ void updateLines(uint2* __restrict__ gl_ptr, const Vec2i* __restrict__ edge, const int num_spring) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_spring; i += blockDim.x * gridDim.x) {
		Vec2i e = edge[i];
		gl_ptr[i].x = (unsigned int)e.x; // todo check if this is needed
		gl_ptr[i].y = (unsigned int)e.y;
	}
}

// update color rgb
__global__ void updateColors(float3* __restrict__ gl_ptr, const Vec3d* __restrict__ color, const int num_mass) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_mass; i += blockDim.x * gridDim.x) {
		Vec3d color_i = color[i];
		gl_ptr[i].x = (float)color_i.x;
		gl_ptr[i].y = (float)color_i.y;
		gl_ptr[i].z = (float)color_i.z;
	}
}

//update triangle indices
__global__ void updateTriangles(uint3* __restrict__ gl_ptr, const Vec3i* __restrict__ triangle, const int num_triangle) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_triangle; i += blockDim.x * gridDim.x) {
		Vec3i triangle_i = triangle[i];
		gl_ptr[i].x = (unsigned int)triangle_i.x;
		gl_ptr[i].y = (unsigned int)triangle_i.y;
		gl_ptr[i].z = (unsigned int)triangle_i.z;
	}
}

void Simulation::updateBuffers() { // todo: check the kernel call
	size_t num_bytes;
	// position update, map OpenGL buffer object for writing from CUDA
	gpuErrchk(cudaGraphicsMapResources(1, &cuda_resource_vertex, stream[CUDA_GRAPHICS_STREAM]));
	cudaGraphicsResourceGetMappedPointer((void**)&dptr_vertex, &num_bytes,cuda_resource_vertex);
	updateVertices << <massBlocksPerGrid, MASS_THREADS_PER_BLOCK, 0, stream[CUDA_GRAPHICS_STREAM] >> > (dptr_vertex, d_mass.pos, mass.size());
	gpuErrchk(cudaGraphicsUnmapResources(1, &cuda_resource_vertex, stream[CUDA_GRAPHICS_STREAM]));// unmap buffer object
	////printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

	// color update
	gpuErrchk(cudaGraphicsMapResources(1, &cuda_resource_color, stream[CUDA_GRAPHICS_STREAM]));
	cudaGraphicsResourceGetMappedPointer((void**)&dptr_color, &num_bytes,cuda_resource_color);
	updateColors << <massBlocksPerGrid, MASS_THREADS_PER_BLOCK, 0, stream[CUDA_GRAPHICS_STREAM] >> > (dptr_color, d_mass.color, mass.size());
	gpuErrchk(cudaGraphicsUnmapResources(1, &cuda_resource_color, stream[CUDA_GRAPHICS_STREAM]));// unmap buffer object

	// line indices update
	gpuErrchk(cudaGraphicsMapResources(1, &cuda_resource_edge, stream[CUDA_GRAPHICS_STREAM]));
	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&dptr_edge, &num_bytes,cuda_resource_edge));
	updateLines << <springBlocksPerGrid, THREADS_PER_BLOCK, 0, stream[CUDA_GRAPHICS_STREAM] >> > (dptr_edge, d_spring.edge, spring.size());
	gpuErrchk(cudaGraphicsUnmapResources(1, &cuda_resource_edge, stream[CUDA_GRAPHICS_STREAM]));// unmap buffer object

	// triangle indices update
	gpuErrchk(cudaGraphicsMapResources(1, &cuda_resource_triangle, stream[CUDA_GRAPHICS_STREAM]));
	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&dptr_triangle, &num_bytes, cuda_resource_triangle));
	updateTriangles << <triangleBlocksPerGrid, THREADS_PER_BLOCK, 0, stream[CUDA_GRAPHICS_STREAM] >> > (dptr_triangle, d_triangle.triangle, triangle.size());
	// unmap buffer object
	gpuErrchk(cudaGraphicsUnmapResources(1, &cuda_resource_triangle, stream[CUDA_GRAPHICS_STREAM]));

}

void Simulation::updateVertexBuffers() {
	size_t num_bytes;
	cudaGraphicsMapResources(1, &cuda_resource_vertex, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&dptr_vertex, &num_bytes,cuda_resource_vertex);
	////printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);
	updateVertices << <massBlocksPerGrid, MASS_THREADS_PER_BLOCK, 0, stream[CUDA_GRAPHICS_STREAM] >> > (dptr_vertex, d_mass.pos, mass.num);
	gpuErrchk(cudaGraphicsUnmapResources(1, &cuda_resource_vertex, 0));// unmap buffer object
}

inline void Simulation::draw() {
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, this->vbo_vertex);
	//glCheckError(); // check opengl error code
	glVertexAttribPointer(
		0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0            // array buffer offset
	);
	//glCheckError(); // check opengl error code
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
	if (show_triangle) {
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_triangle);
		glDrawElements(GL_TRIANGLES, 3 * triangle.size(), GL_UNSIGNED_INT, (void*)0); // 2 indices for a line
	}
	
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(0);
}


void Simulation::framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

void Simulation::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{ // ref: https://www.glfw.org/docs/latest/input_guide.html#input_key
	if (key == GLFW_KEY_F && action == GLFW_PRESS)
	{
		const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
		// if window is maximized
		if (glfwGetWindowAttrib(window, GLFW_MAXIMIZED)) {//restore default window size
			glfwRestoreWindow(window);
			glfwSetWindowAttrib(window, GLFW_DECORATED, GL_TRUE);
			glfwSetWindowMonitor(window, NULL, 50, 50, 0.5*mode->width,0.5*mode->height, GLFW_DONT_CARE);
		}
		else { // maximize the window,fullscreen
			glfwMaximizeWindow(window);
			glfwSetWindowAttrib(window, GLFW_MAXIMIZED, GL_TRUE);
			glfwSetWindowAttrib(window, GLFW_DECORATED, GL_FALSE);
			glfwSetWindowMonitor(window, NULL, 0, 0, mode->width, mode->height, GLFW_DONT_CARE);
		}
	}
	else if (key == GLFW_KEY_T && action == GLFW_PRESS) {//enable/disable showing triangle faces
		Simulation* sim = (Simulation*)glfwGetWindowUserPointer(window);
		sim->show_triangle = !sim->show_triangle;
	}
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
	glfwWindowHint(GLFW_AUTO_ICONIFY, GLFW_FALSE);//simultaneously cover multiple windows with full screen windows

	auto monitor = glfwGetPrimaryMonitor();
	const GLFWvidmode* mode = glfwGetVideoMode(monitor);
	glfwWindowHint(GLFW_RED_BITS, mode->redBits);
	glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
	glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
	glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
	//window = glfwCreateWindow(mode->width, mode->height, "CUDA Physics Simulation", glfwGetPrimaryMonitor(), NULL);
	glfwWindowHint(GLFW_DECORATED, GL_TRUE);//enable/disable close widget, etc, for fake full screen effect
	//window = glfwCreateWindow(mode->width, mode->height+1, "CUDA Physics Simulation", NULL, NULL);//+1 to fake full screen
	window = glfwCreateWindow(1920, 1080, "CUDA Physics Simulation", NULL, NULL);
	// set pointer to pass parameters for callbacks
	glfwSetWindowUserPointer(window, this);//ref: https://discourse.glfw.org/t/passing-parameters-to-callbacks/848/2

	if (window == NULL) {
		fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 4.6 compatible.\n");
		getchar();
		glfwTerminate();
		exit(1);
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetKeyCallback(window, key_callback);

	glEnable(GL_CULL_FACE); // FACE CULLING :https://learnopengl.com/Advanced-OpenGL/Face-culling
	glCullFace(GL_BACK);
	glFrontFace(GL_CCW);

	//glShadeModel(GL_FLAT); //flat shading
	//glShadeModel(GL_SMOOTH); //smooth shading
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
