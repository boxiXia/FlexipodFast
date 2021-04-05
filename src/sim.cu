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



#ifdef STRESS_TEST
__global__ void updtateSpringStrain(
	const MASS mass,
	const SPRING spring,
	double* strain,//selected strains (deformation/rest_length) of the spring
	const int start,//start index (inclusive), must<spring.size()
	const int end, // end index (exclusive), must<=spring.size()
	const int step // step size
) {
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	int i = start + k * step;
	if (i < end) {
		Vec2i e = spring.edge[i];
		Vec3d s_vec = mass.pos[e.y] - mass.pos[e.x];// the vector from left to right
		double length = s_vec.norm(); // current spring length
		strain[i] = (length - spring.rest[i]) / spring.rest[i];
	}
}
#endif //STRESS_TEST



__global__ void updateSpring(
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

__global__ void updateSpringAndReset(
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


__global__ void updateMass(
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

// roate the mass of the joint directly
__global__ void updateJoint(Vec3d* __restrict__ mass_pos, const JOINT joint) {
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


#ifdef GRAPHICS
constexpr int NUM_PER_VERTEX = 9; // number of float per vertex;
constexpr int VERTEX_COLOR_OFFSET = 3; // offset for the vertex color
constexpr int VERTEX_NORMAL_OFFSET = 6; // offset for the vertex normal
// vertex gl buffer: x,y,z,r,g,b,nx,ny,nz

// update vertex positions
__global__ void updateVertices(GLfloat* __restrict__ gl_ptr, const Vec3d* __restrict__  pos, const int num_mass) {
	// https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
	// https://devblogs.nvidia.com/cuda-pro-tip-optimize-pointer-aliasing/
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_mass; i += blockDim.x * gridDim.x) {
		Vec3d pos_i = pos[i];
		int k = NUM_PER_VERTEX * i; //x,y,z
		// update positions
		gl_ptr[k] = (GLfloat)pos_i.x;
		gl_ptr[k + 1] = (GLfloat)pos_i.y;
		gl_ptr[k + 2] = (GLfloat)pos_i.z;
		// zero vertex normals, must run before update normals
		k += VERTEX_NORMAL_OFFSET;
		gl_ptr[k] = 0.f;
		gl_ptr[k + 1] = 0.f;
		gl_ptr[k + 2] = 0.f;
	}
}

// update color rgb
__global__ void updateColors(GLfloat* __restrict__ gl_ptr, const Vec3d* __restrict__ color, const int num_mass) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_mass; i += blockDim.x * gridDim.x) {
		Vec3d color_i = color[i];
		int k = NUM_PER_VERTEX * i + 3; // x,y,z, r,g,b
		gl_ptr[k] = (GLfloat)color_i.x;
		gl_ptr[k + 1] = (GLfloat)color_i.y;
		gl_ptr[k + 2] = (GLfloat)color_i.z;
	}
}

// update vertex normals from triangles, should run after updateVertices
__global__ void updateTriangleVertexNormal(
	GLfloat* __restrict__ gl_ptr,
	const Vec3d* __restrict__  pos, // vertex positions
	const Vec3i* __restrict__ triangle, // triangle indices
	const int num_triangles
) { // https://www.iquilezles.org/www/articles/normals/normals.htm
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_triangles; i += blockDim.x * gridDim.x) {
		Vec3i t = triangle[i]; // triangle indices i
		//TODO check if direction is correct
		Vec3d pos_ty = pos[t.y];
		Vec3d e1 = pos_ty - pos[t.x];
		Vec3d e2 = pos[t.z] - pos_ty;
		Vec3d no = cross(e1, e2); // triangle normals

		float nx = (float)no.x;
		float ny = (float)no.y;
		float nz = (float)no.z;

		int k = NUM_PER_VERTEX * t.x + VERTEX_NORMAL_OFFSET; // x,y,z, r,g,b,nx,ny,nz
		atomicAdd(&gl_ptr[k], nx);
		atomicAdd(&gl_ptr[k + 1], ny);
		atomicAdd(&gl_ptr[k + 2], nz);
		//gl_ptr[k] += nx;
		//gl_ptr[k + 1] += ny;
		//gl_ptr[k + 2] += nz;

		k = NUM_PER_VERTEX * t.y + VERTEX_NORMAL_OFFSET; // x,y,z, r,g,b,nx,ny,nz
		atomicAdd(&gl_ptr[k], nx);
		atomicAdd(&gl_ptr[k + 1], ny);
		atomicAdd(&gl_ptr[k + 2], nz);
		//gl_ptr[k] += nx;
		//gl_ptr[k + 1] += ny;
		//gl_ptr[k + 2] += nz;

		k = NUM_PER_VERTEX * t.z + VERTEX_NORMAL_OFFSET; // x,y,z, r,g,b,nx,ny,nz
		atomicAdd(&gl_ptr[k], nx);
		atomicAdd(&gl_ptr[k + 1], ny);
		atomicAdd(&gl_ptr[k + 2], nz);
		//gl_ptr[k] += nx;
		//gl_ptr[k + 1] += ny;
		//gl_ptr[k + 2] += nz;
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

//update triangle indices
__global__ void updateTriangles(uint3* __restrict__ gl_ptr, const Vec3i* __restrict__ triangle, const int num_triangle) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_triangle; i += blockDim.x * gridDim.x) {
		Vec3i triangle_i = triangle[i];
		gl_ptr[i].x = (unsigned int)triangle_i.x;
		gl_ptr[i].y = (unsigned int)triangle_i.y;
		gl_ptr[i].z = (unsigned int)triangle_i.z;
	}
}

#endif


//Simulation::Simulation() {
//	//cudaSetDevice(1);
//	for (int i = 0; i < NUM_CUDA_STREAM; ++i) { // lower i = higher priority
//		cudaStreamCreateWithPriority(&stream[i], cudaStreamDefault, i);// create extra cuda stream: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution
//	}
//}

Simulation::Simulation(size_t num_mass, size_t num_spring, size_t num_joint, size_t num_triangle) :
	mass(num_mass, true), // allocate host
	d_mass(num_mass, false),// allocate device
	spring(num_spring, true),// allocate host
	d_spring(num_spring, false),// allocate device
	triangle(num_triangle, true),//allocate host
	d_triangle(num_triangle, false),//allocate device
	joint_control(num_joint, true) // joint controller, must also call reset, see updatePhysics()
#ifdef UDP
	, udp_server(port_local, port_remote, ip_remote)// port_local,port_remote,ip_remote,num_joint
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

//int Simulation::computeBlocksPerGrid(const int threadsPerBlock, const int num) {
//	int blocksPerGrid = (num - 1 + threadsPerBlock) / threadsPerBlock;
//	assert(blocksPerGrid <= MAX_BLOCKS);//TODO: kernel has a hard limit on MAX_BLOCKS
//	return blocksPerGrid;
//}

/* compute the block size (threads per block) and grid size （blocks per grid)
*
*/
int Simulation::computeGridSize(int block_size, int num) {
	int grid_size = (num - 1 + block_size) / block_size;// Round up according to array size 
	assert(grid_size <= MAX_BLOCKS);//TODO: kernel has a hard limit on MAX_BLOCKS
	return grid_size;
}

void Simulation::updateCudaParameters() {
	// ref: https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
	int minGridSize;
	//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &spring_block_size, updateSpring, 0, 0);
	spring_grid_size = computeGridSize(spring_block_size, spring.size());

	//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &mass_block_size, updateMass, 0, 0);
	mass_grid_size = computeGridSize(mass_block_size, mass.size());

	//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &joint_block_size, updateJoint, 0, 0);
	joint_grid_size = computeGridSize(joint_block_size, joint.points.size());

#ifdef GRAPHICS
	//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &triangle_block_size, updateTriangleVertexNormal, 0, 0);
	triangle_grid_size = computeGridSize(triangle_block_size, triangle.size());

	//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &vertex_block_size, updateVertices, 0, 0);
	vertex_grid_size = computeGridSize(vertex_block_size, mass.size());
#endif //GRAPHICS

}

void Simulation::setBreakpoint(const double time, const bool should_end) {
	//assert(!ENDED, "Simulation has ended. Cannot setBreakpoint.");
	bpts.insert(BreakPoint(time, should_end)); // TODO mutex breakpoints
}

/*pause the simulation at (simulation) time t [s] */
void Simulation::pause(const double t) {
	assert(!ENDED, "Simulation has ended. can't call pause");
	setBreakpoint(t, false);
	//// Wait until simulation is actually paused
	std::unique_lock<std::mutex> lck(mutex_running);
	SHOULD_RUN = false;
	cv_running.notify_all();
	cv_running.wait(lck, [this] {return !RUNNING; });
}


void Simulation::resume() {
	assert(!ENDED, "Simulation has ended. Cannot resume simulation.");
	assert(STARTED, "Simulation has not started. Cannot resume before calling sim.start().");
	assert(mass.num > 0, "No masses have been added. Add masses before simulation starts.");
	//updateCudaParameters();
	//cudaDeviceSynchronize();
	std::unique_lock<std::mutex> lck(mutex_running);
	SHOULD_RUN = true;
	cv_running.notify_all();
}

void Simulation::waitForEvent() {
	assert(!ENDED, "Simulation has ended. can't call waitForEvent()");
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
	cudaDeviceSynchronize();
	d_mass.copyFrom(backup_mass, stream[CUDA_MEMORY_STREAM]);
	mass.copyFrom(backup_mass, stream[CUDA_MEMORY_STREAM_ALT]);

	d_spring.copyFrom(backup_spring, stream[CUDA_MEMORY_STREAM]);
	spring.copyFrom(backup_spring, stream[CUDA_MEMORY_STREAM_ALT]);

	d_joint.copyFrom(backup_joint, stream[CUDA_MEMORY_STREAM]);
	joint.copyFrom(backup_joint, stream[CUDA_MEMORY_STREAM_ALT]);

	joint_control.reset(backup_mass, backup_joint);
	//joint_control.update(backup_mass, backup_joint, dt);
	body.init(backup_mass, id_oxyz_start); // init body frame
	cudaDeviceSynchronize();
	RESET = false; // set reset to false 
}


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

	setAll();// copy mass and spring to gpu

	backupState();// backup the robot mass/spring/joint state

#ifdef UDP
	udp_server.run();
	thread_msg_update = std::thread(&Simulation::updateUdpMessage, this); //TODO: thread
#endif //UDP

#ifdef GRAPHICS
	thread_graphics_update = std::thread(&Simulation::updateGraphics, this);
#endif// Graphics
	thread_physics_update = std::thread(&Simulation::updatePhysics, this);

	{
		int device;
		cudaGetDevice(&device);
		printf("cuda device: %d\n", device);
	}

}

void Simulation::updatePhysics() { // repeatedly start next

	//cudaDeviceProp deviceProp;
	//cudaGetDeviceProperties(&deviceProp, 0);
	//if (deviceProp.cooperativeLaunch == 0) {
	//	printf("not supported");
	//	exit(-1);
	//}


#ifdef STRESS_TEST

#endif //STRESS_TEST



	cudaDeviceSynchronize();//sync before while loop
	auto start = std::chrono::steady_clock::now();
	int k_rot = 0; // rotation counter

#ifdef DEBUG_ENERGY
	energy_start = energy(); // compute the total energy of the system at T=0
#endif // DEBUG_ENERGY

	joint_control.reset(mass, joint);// reset joint controller, must do
	body.init(mass, id_oxyz_start); // init body frame

	while (true) {
		if (!bpts.empty() && (*bpts.begin()).t <= T) {// paused when a break p
			//cudaDeviceSynchronize(); // synchronize before updating the springs and mass positions
		//            std::cout << "Breakpoint set for time " << *bpts.begin() << " reached at simulation time " << T << "!" << std::endl;

			do {
				SHOULD_END = (*bpts.begin()).should_end;
				bpts.erase(bpts.begin());// remove all breakpoints <= T
			} while (!bpts.empty() && (*bpts.begin()).t <= T);

			{	// condition variable 
				std::unique_lock<std::mutex> lck(mutex_running); // refer to:https://en.cppreference.com/w/cpp/thread/condition_variable
				RUNNING = false;
				SHOULD_RUN = false;
				lck.unlock();// Manual unlocking before notifying, to avoid waking up the waiting thread only to block again
				cv_running.notify_all(); //notify others RUNNING = false

				std::chrono::steady_clock::time_point ct_begin = std::chrono::steady_clock::now();
				while (!(SHOULD_RUN || SHOULD_END)) { // paused
#ifdef UDP
					bool msg_received = ReceiveUdpMessage();
					// send message every 2 ms or received new message (500Hz)
					std::chrono::steady_clock::time_point ct_end = std::chrono::steady_clock::now();
					float diff = std::chrono::duration_cast<std::chrono::milliseconds>(ct_end - ct_begin).count();
					if (diff >= 2 || msg_received) {
						udp_server.send();// send udp message
						ct_begin = ct_end;
					}
#endif // UDP
				}
				lck.lock();
				//cv_running.wait(lck, [this] {return SHOULD_RUN||SHOULD_END; }); // wait unitl SHOULD_RUN is signaled
				RUNNING = true;
				lck.unlock();
				cv_running.notify_all(); // notifiy others RUNNING = true;
			}
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

#ifdef GRAPHICS
				{
					std::unique_lock<std::mutex> lck(mutex_running); // refer to:https://en.cppreference.com/w/cpp/thread/condition_variable
					cv_running.wait(lck, [this] {return GRAPHICS_ENDED; });
				}
#endif //GRAPHICS
#ifdef UDP
				{
					std::unique_lock<std::mutex> lck(mutex_running);
					cv_running.wait(lck, [this] {return UDP_ENDED; });
				}
#endif //UDP
				RUNNING = false;
				ENDED = true; // TODO maybe race condition
				cv_running.notify_all(); //notify others RUNNING = false
				return;
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


#ifdef UDP
		ReceiveUdpMessage(); // receive udp message whenever there is new
#endif // UDP

		joint_control.update(mass, joint, NUM_QUEUED_KERNELS * dt);
		// update joint speed

		for (int i = 0; i < joint.anchors.num; i++) { // compute joint angles and angular velocity
			joint.anchors.theta[i] = NUM_UPDATE_PER_ROTATION * joint_control.cmd[i] * dt;// update joint speed
		}
		d_joint.anchors.copyThetaFrom(joint.anchors, stream[CUDA_DYNAMICS_STREAM]);

		// invoke cuda kernel for dynamics update
		for (int i = 0; i < NUM_QUEUED_KERNELS; i++) {
#ifdef ROTATION
			if (k_rot % NUM_UPDATE_PER_ROTATION == 0) {
				k_rot = 0; // reset counter
				updateJoint << <joint_grid_size, joint_block_size, 0, stream[CUDA_DYNAMICS_STREAM] >> > (d_mass.pos, d_joint);
				updateSpringAndReset << <spring_grid_size, spring_block_size, 0, stream[CUDA_DYNAMICS_STREAM] >> > (d_mass, d_spring);
			}
			else {
				updateSpring << <spring_grid_size, spring_block_size, 0, stream[CUDA_DYNAMICS_STREAM] >> > (d_mass, d_spring);
			}
#else
			updateSpring << <spring_grid_size, spring_block_size, 0, stream[CUDA_DYNAMICS_STREAM] >> > (d_mass, d_spring);
#endif // ROTATION
			updateMass << <mass_grid_size, mass_block_size, 0, stream[CUDA_DYNAMICS_STREAM] >> > (d_mass, d_constraints, global_acc, dt);
			//gpuErrchk(cudaPeekAtLastError());
			k_rot++;
		}

		T += NUM_QUEUED_KERNELS * dt;

		//mass.CopyPosVelAccFrom(d_mass, stream[CUDA_DYNAMICS_STREAM]);
		mass.CopyPosFrom(d_mass, stream[CUDA_DYNAMICS_STREAM]);
		cudaStreamSynchronize(stream[CUDA_DYNAMICS_STREAM]);

		//cudaDeviceSynchronize();

		//std::chrono::steady_clock::time_point ct_begin = std::chrono::steady_clock::now();
		////std::chrono::steady_clock::time_point ct_end = std::chrono::steady_clock::now();
		////std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (ct_end - ct_begin).count() << "[ns]" << std::endl;


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
#ifdef UDP
		SHOULD_SEND_UDP = true;
#endif // UDP

		// restore the robot mass/spring/joint state to the backedup state
		if (RESET) { resetState(); }
	}
}


#ifdef UDP
bool Simulation::ReceiveUdpMessage() {
	// receiving message
	if (udp_server.flag_new_received) {
		udp_server.flag_new_received = false;
		switch (udp_server.msg_rec.header) {
		case UDP_HEADER::TERMINATE://close the program
			bpts.insert(BreakPoint(0, true));//SHOULD_END = true;
			break;
		case UDP_HEADER::RESET: // reset
			// set the reset flag to true, resetState() will be called 
			//RESET = true;// to restore mass/spring/joint state to the backedup state
			SHOULD_RUN = true;
			resetState();// restore the robot mass/spring/joint state to the backedup state
			break;
		case UDP_HEADER::MOTOR_SPEED_COMMEND:
			joint_control.updateControlMode(JointControlMode::vel);
			for (int i = 0; i < joint_control.size(); i++) {//update joint speed from received udp packet
				joint_control.vel_desired[i] = udp_server.msg_rec.joint_value_desired[i];
			}
			break;
		case UDP_HEADER::MOTOR_POS_COMMEND:
			joint_control.updateControlMode(JointControlMode::pos);
			for (int i = 0; i < joint_control.size(); i++) {//update joint speed from received udp packet
				joint_control.pos_desired[i] = udp_server.msg_rec.joint_value_desired[i];
			}
			break;
		case UDP_HEADER::PAUSE:
			bpts.insert(BreakPoint(0, false));//SHOULD_END = true;
			break;
		case UDP_HEADER::RESUME:
			SHOULD_RUN = true;
			break;
		case UDP_HEADER::STEP_MOTOR_SPEED_COMMEND:
			joint_control.updateControlMode(JointControlMode::vel);
			for (int i = 0; i < joint_control.size(); i++) {//update joint speed from received udp packet
				joint_control.vel_desired[i] = udp_server.msg_rec.joint_value_desired[i];
			}
			if (!RUNNING) { SHOULD_RUN = true; }
			setBreakpoint(T + NUM_QUEUED_KERNELS * dt * NUM_UDP_MULTIPLIER);
			break;
		default:
			break;
		}
		return true;
	}
	else { return false; }
}

void Simulation::updateUdpMessage() {
	while (!SHOULD_END) {
		if (SHOULD_SEND_UDP) {
			SHOULD_SEND_UDP = false;

			body.update(mass, id_oxyz_start, NUM_QUEUED_KERNELS * dt);
			udp_server.msg_send.emplace_front(
				DataSend(UDP_HEADER::ROBOT_STATE_REPORT, T, joint_control, body
#ifdef STRESS_TEST	
					, id_part_end, mass, spring
#endif //STRESS_TEST
				));

			if (udp_server.msg_send.size() > NUM_UDP_MULTIPLIER) {
				udp_server.msg_send.pop_back();
			}
			udp_server.send();// send udp message
		}
		//ReceiveUdpMessage();// receiving message
	}
	{	// notify the physics thread // https://en.cppreference.com/w/cpp/thread/condition_variable
		std::lock_guard<std::mutex> lk(mutex_running);
		UDP_ENDED = true;
	}
	cv_running.notify_all();
}

#endif //UDP



#ifdef GRAPHICS
void Simulation::updateGraphics() {

	createGLFWWindow(); // create a window with  width and height

	int glDeviceId;// todo:this output wrong number, maybe it is a cuda bug...
	unsigned int glDeviceCount;
	cudaGLGetDevices(&glDeviceCount, &glDeviceId, 1u, cudaGLDeviceListAll);
	printf("openGL device: %u\n", glDeviceId);

	glGenVertexArrays(1, &VertexArrayID);//GLuint VertexArrayID;
	glBindVertexArray(VertexArrayID);

	// get the directory of this program
	std::string program_dir = getProgramDir();
	// Create and compile our GLSL program from the shaders
	shader = Shader(
		program_dir + "\\shaderVertex.glsl",
		program_dir + "\\shaderFragment.glsl");
	//shader.use();

	//https://learnopengl.com/Advanced-Lighting/Shadows/Point-Shadows
	//https://github.com/JoeyDeVries/LearnOpenGL/tree/master/src/5.advanced_lighting/3.1.3.shadow_mapping
	simpleDepthShader = Shader(
		program_dir + "\\shadow_mapping_depth_vertex.glsl",
		program_dir + "\\shadow_mapping_depth_fragment.glsl");
	//simpleDepthShader.use();

	/*------------------- configure depth map FBO ----------------------------*/
	glGenFramebuffers(1, &depthMapFBO);
	// create depth texture
	glGenTextures(1, &depthMap);
	glBindTexture(GL_TEXTURE_2D, depthMap);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	float borderColor[] = { 1.0, 1.0, 1.0, 1.0 };
	glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
	// attach depth texture as FBO's depth buffer
	glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);
	glDrawBuffer(GL_NONE);
	glReadBuffer(GL_NONE);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	/*-------------------------------------------------------------------------*/

	computeMVP(); // compute perspective projection matrix

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

	while (!SHOULD_END) {

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

			// copy only the position
			//mass.CopyPosFrom(d_mass, stream[CUDA_MEMORY_STREAM]);
			//mass.CopyPosFrom(d_mass, id_oxyz_start,1,stream[CUDA_MEMORY_STREAM]);
			Vec3d com_pos = mass.pos[id_oxyz_start];// center of mass position (anchored body center)

			double t_lerp = 0.02;
			double speed_multiplier = 0.1;
			double pos_multiplier = 0.05;

			if (glfwGetKey(window, GLFW_KEY_UP)) {
				for (int i = 0; i < joint.size(); i++) {
					if (joint_control.mode == vel) {
						joint_control.vel_desired[i] += i < 2 ? speed_multiplier : -speed_multiplier;
					}
					else if (joint_control.mode == pos) {
						joint_control.pos_desired[i] += i < 2 ? pos_multiplier : -pos_multiplier;
					}
				}
			}
			else if (glfwGetKey(window, GLFW_KEY_DOWN)) {
				for (int i = 0; i < joint.size(); i++) {
					if (joint_control.mode == vel) {
						joint_control.vel_desired[i] -= i < 2 ? speed_multiplier : -speed_multiplier;
					}
					else if (joint_control.mode == pos) {
						joint_control.pos_desired[i] -= i < 2 ? pos_multiplier : -pos_multiplier;
					}
				}
			}
			if (glfwGetKey(window, GLFW_KEY_LEFT)) {
				for (int i = 0; i < joint.size(); i++) {
					if (joint_control.mode == vel) {
						joint_control.vel_desired[i] -= speed_multiplier;
					}
					else if (joint_control.mode == pos) {
						joint_control.pos_desired[i] -= pos_multiplier;
					}
				}
			}
			else if (glfwGetKey(window, GLFW_KEY_RIGHT)) {
				for (int i = 0; i < joint.size(); i++) {
					if (joint_control.mode == vel) {
						joint_control.vel_desired[i] += speed_multiplier;
					}
					else if (joint_control.mode == pos) {
						joint_control.pos_desired[i] += pos_multiplier;
					}
				}
			}
			else if (glfwGetKey(window, GLFW_KEY_0)) { // zero speed
				joint_control.reset(mass, joint);
			}


			// https://en.wikipedia.org/wiki/Slerp
			//mass.pos[id_oxyz_start].print();


			Vec3d camera_rotation_anchor = com_pos + (camera_up_offset - com_pos.dot(camera_up)) * camera_up;

			Vec3d camera_pos_desired = AxisAngleRotaion(camera_up, camera_href_dir * camera_h_offset, camera_yaw, Vec3d()) + camera_rotation_anchor;

			camera_pos = lerp(camera_pos, camera_pos_desired, t_lerp);

			Vec3d camera_dir_desired = (com_pos - camera_pos).normalize();

			// spherical linear interpolation from camera_dir to camera_dir_new by factor t_lerp
			camera_dir = slerp(camera_dir, camera_dir_desired, t_lerp).normalize();

			computeMVP(true); // update MVP, also update camera matrix //todo

			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear screen


			// 1. render depth of scene to texture (from light's perspective)
			// --------------------------------------------------------------
			float near_plane = -5.0, far_plane = 10.f;
			//lightProjection = glm::perspective(glm::radians(45.0f), (GLfloat)SHADOW_WIDTH / (GLfloat)SHADOW_HEIGHT, near_plane, far_plane); // note that if you use a perspective projection matrix you'll have to change the light position as the current light position isn't enough to reflect the whole scene
			glm::mat4 lightProjection = glm::ortho(-1.5f, 1.5f, -1.5f, 1.5f, near_plane, far_plane);

			//lightView = glm::lookAt(light.direction, glm::vec3(0.0f), glm::vec3(0.0, 1.0, 0.0));
			glm::vec3 light_pos = glm::vec3(
				com_pos.x + light.direction.x,
				com_pos.y + light.direction.y,
				com_pos.z + light.direction.z);

			glm::mat4 lightView = glm::lookAt(
				light_pos, // camera position in World Space
				glm::vec3(com_pos.x, com_pos.y, com_pos.z),// look at position
				glm::vec3(0., 1.0, 0.0));  // camera up vector (set to 0,-1,0 to look upside-down)

			glm::mat4 lightSpaceMatrix = lightProjection * lightView;
			// render scene from light's point of view
			simpleDepthShader.use();
			simpleDepthShader.setMat4("lightSpaceMatrix", lightSpaceMatrix);

			glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
			glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
			glClear(GL_DEPTH_BUFFER_BIT);
			draw();
			glBindFramebuffer(GL_FRAMEBUFFER, 0);

			// reset viewport
			glViewport(0, 0, framebuffer_width, framebuffer_height);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


			/*---- 2. render scene as normal using the generated depth/shadow map--*/
			shader.use(); // use the shader
			light.set(shader.ID, "light"); // set the light uniform
			shader.setMat4("MVP", MVP); // set MVP
			shader.setVec3("viewPos", // set view position
				(float)camera_pos.x, (float)camera_pos.y, (float)camera_pos.z);
			shader.setMat4("lightSpaceMatrix", lightSpaceMatrix);
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, depthMap);
			draw();
			/*--------------------------------------------------------------------*/

			// Swap buffers, render screen
			glfwPollEvents();
			glfwSwapBuffers(window);

			//// check for errors
			//GLenum error = glGetError();
			//if (error != GL_NO_ERROR)
			//	std::cerr << "OpenGL Error " << error << std::endl;

			if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS || glfwWindowShouldClose(window) != 0) {
				bpts.insert(BreakPoint(T, true));// break at current time T // TODO maybe deal with memory leak here. //key press exit,
				// condition variable: REF: https://en.cppreference.com/w/cpp/thread/condition_variable
				std::lock_guard<std::mutex> lk(mutex_running);
				SHOULD_END = true;
				cv_running.notify_all(); //notify others SHOULD_END = true
			}
		}

	}

	// end the graphics
	deleteBuffers(); // delete the buffer objects
	glDeleteProgram(shader.ID);
	glDeleteVertexArrays(1, &VertexArrayID);
	glfwTerminate(); // Close OpenGL window and terminate GLFW
	printf("\nwindow closed\n");

	{	// notify the physics thread // https://en.cppreference.com/w/cpp/thread/condition_variable
		std::lock_guard<std::mutex> lk(mutex_running);
		GRAPHICS_ENDED = true;
	}
	cv_running.notify_all();


	return;


}
#endif




Simulation::~Simulation() {
	std::cout << "Simulation destructor called." << std::endl;

	if (STARTED) {
		{
			std::unique_lock<std::mutex> lck(mutex_running);//https://en.cppreference.com/w/cpp/thread/condition_variable
			cv_running.wait(lck, [this] {return ENDED; });
		}
		assert(thread_physics_update.joinable());
		thread_physics_update.join();
		//printf("thread_physics_update joined\n");

#ifdef GRAPHICS
		assert(thread_graphics_update.joinable());
		thread_graphics_update.join();
		//printf("thread_graphics_update joined\n");
#endif

#ifdef UDP
		assert(thread_msg_update.joinable());
		thread_msg_update.join();
		udp_server.close();
#endif // UDP
	}
	freeGPU();
	printf("Simulation ended\n");
}


void Simulation::freeGPU() {
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
* ref:https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OPENGL.html
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
	createVBO(&vbo_vertex, &cuda_resource_vertex, mass.size() * sizeof(GLfloat) * NUM_PER_VERTEX, cudaGraphicsRegisterFlagsNone, GL_ARRAY_BUFFER);//TODO CHANGE TO WRITE ONLY
	createVBO(&vbo_edge, &cuda_resource_edge, spring.size() * sizeof(uint2), cudaGraphicsRegisterFlagsNone, GL_ELEMENT_ARRAY_BUFFER);
	createVBO(&vbo_triangle, &cuda_resource_triangle, triangle.size() * sizeof(uint3), cudaGraphicsRegisterFlagsNone, GL_ELEMENT_ARRAY_BUFFER);
}

inline void Simulation::resizeBuffers() {
	////TODO>>>>>>
	resizeVBO(&vbo_vertex, mass.size() * sizeof(GLfloat) * NUM_PER_VERTEX, GL_ARRAY_BUFFER);//TODO CHANGE TO WRITE ONLY
	resizeVBO(&vbo_edge, spring.size() * sizeof(uint2), GL_ELEMENT_ARRAY_BUFFER);
	resizeVBO(&vbo_triangle, triangle.size() * sizeof(uint3), GL_ELEMENT_ARRAY_BUFFER);
	resize_buffers = false;
}
inline void Simulation::deleteBuffers() {
	deleteVBO(&vbo_vertex, cuda_resource_vertex, GL_ARRAY_BUFFER);
	deleteVBO(&vbo_edge, cuda_resource_edge, GL_ELEMENT_ARRAY_BUFFER);
	deleteVBO(&vbo_triangle, cuda_resource_triangle, GL_ELEMENT_ARRAY_BUFFER);
}


void Simulation::updateBuffers() { // todo: check the kernel call
	size_t num_bytes;
	// vertex update, map OpenGL buffer object for writing from CUDA: update positions and colors
	gpuErrchk(cudaGraphicsMapResources(1, &cuda_resource_vertex, stream[CUDA_GRAPHICS_STREAM]));
	cudaGraphicsResourceGetMappedPointer((void**)&dptr_vertex, &num_bytes, cuda_resource_vertex);
	updateVertices << <vertex_grid_size, vertex_block_size, 0, stream[CUDA_GRAPHICS_STREAM] >> > (dptr_vertex, d_mass.pos, mass.size());
	updateColors << <vertex_grid_size, vertex_block_size, 0, stream[CUDA_GRAPHICS_STREAM] >> > (dptr_vertex, d_mass.color, mass.size());
	updateTriangleVertexNormal << <triangle_grid_size, triangle_block_size, 0, stream[CUDA_GRAPHICS_STREAM] >> > (dptr_vertex, d_mass.pos, d_triangle.triangle, triangle.size());

	gpuErrchk(cudaGraphicsUnmapResources(1, &cuda_resource_vertex, stream[CUDA_GRAPHICS_STREAM]));// unmap buffer object
	////printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

	// line indices update
	gpuErrchk(cudaGraphicsMapResources(1, &cuda_resource_edge, stream[CUDA_GRAPHICS_STREAM]));
	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&dptr_edge, &num_bytes, cuda_resource_edge));
	updateLines << <spring_grid_size, spring_block_size, 0, stream[CUDA_GRAPHICS_STREAM] >> > (dptr_edge, d_spring.edge, spring.size());
	gpuErrchk(cudaGraphicsUnmapResources(1, &cuda_resource_edge, stream[CUDA_GRAPHICS_STREAM]));// unmap buffer object

	// triangle indices update
	gpuErrchk(cudaGraphicsMapResources(1, &cuda_resource_triangle, stream[CUDA_GRAPHICS_STREAM]));
	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&dptr_triangle, &num_bytes, cuda_resource_triangle));
	updateTriangles << <triangle_grid_size, triangle_block_size, 0, stream[CUDA_GRAPHICS_STREAM] >> > (dptr_triangle, d_triangle.triangle, triangle.size());
	gpuErrchk(cudaGraphicsUnmapResources(1, &cuda_resource_triangle, stream[CUDA_GRAPHICS_STREAM]));// unmap buffer object

}

void Simulation::updateVertexBuffers() {
	size_t num_bytes;
	cudaGraphicsMapResources(1, &cuda_resource_vertex, stream[CUDA_GRAPHICS_STREAM]);
	cudaGraphicsResourceGetMappedPointer((void**)&dptr_vertex, &num_bytes, cuda_resource_vertex);
	////printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);
	updateVertices << <vertex_grid_size, vertex_block_size, 0, stream[CUDA_GRAPHICS_STREAM] >> > (dptr_vertex, d_mass.pos, mass.num);
	if (show_triangle) {
		updateTriangleVertexNormal << <triangle_grid_size, triangle_block_size, 0, stream[CUDA_GRAPHICS_STREAM] >> > (dptr_vertex, d_mass.pos, d_triangle.triangle, triangle.size());
	}
	cudaGraphicsUnmapResources(1, &cuda_resource_vertex, stream[CUDA_GRAPHICS_STREAM]);// unmap buffer object
}

inline void Simulation::draw() {
	// ref: https://stackoverflow.com/questions/16380005/opengl-3-4-glvertexattribpointer-stride-and-offset-miscalculation
	glBindBuffer(GL_ARRAY_BUFFER, this->vbo_vertex);
	//glCheckError(); // check opengl error code
	glVertexAttribPointer( // vertex position
		0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		NUM_PER_VERTEX * sizeof(GL_FLOAT),// stride
		(void*)0            // array buffer offset
	);
	glEnableVertexAttribArray(0);

	//glCheckError(); // check opengl error code
	glVertexAttribPointer( // vertex color
		1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
		3,                                // size
		GL_FLOAT,                         // type
		GL_FALSE,                         // normalized?
		NUM_PER_VERTEX * sizeof(GL_FLOAT),   // stride
		(GLvoid*)(VERTEX_COLOR_OFFSET * sizeof(GL_FLOAT))   // array buffer offset
	);
	glEnableVertexAttribArray(1);

	glVertexAttribPointer( // vertex normal
		2,                                // attribute. No particular reason for 1, but must match the layout in the shader.
		3,                                // size
		GL_FLOAT,                         // type
		GL_FALSE,                         // normalized?
		NUM_PER_VERTEX * sizeof(GL_FLOAT),   // stride
		(GLvoid*)(VERTEX_NORMAL_OFFSET * sizeof(GL_FLOAT))   // array buffer offset
	);
	glEnableVertexAttribArray(2);

	if (show_triangle) {
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_triangle);
		glDrawElements(GL_TRIANGLES, 3 * triangle.size(), GL_UNSIGNED_INT, (void*)0); // 3 indices for a triangle
	}
	else { // draw lines
		glDrawArrays(GL_POINTS, 0, mass.num); // 3 indices starting at 0 -> 1 triangle
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_edge);
		glDrawElements(GL_LINES, 2 * spring.num, GL_UNSIGNED_INT, (void*)0); // 2 indices for a line
	}

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);

	// draw constraints
	for (Constraint* c : constraints) {
		c->draw();
	}
}


void Simulation::framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

void Simulation::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{ // ref: https://www.glfw.org/docs/latest/input_guide.html#input_key
	Simulation& sim = *(Simulation*)glfwGetWindowUserPointer(window);
	if (action == GLFW_PRESS) {
		if (key == GLFW_KEY_F) {
			const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
			// if window is maximized
			if (glfwGetWindowAttrib(window, GLFW_MAXIMIZED)) {//restore default window size
				glfwRestoreWindow(window);
				glfwSetWindowAttrib(window, GLFW_DECORATED, GL_TRUE);
				glfwSetWindowMonitor(window, NULL, 50, 50, int(0.5 * mode->width), int(0.5 * mode->height), GLFW_DONT_CARE);
			}
			else { // maximize the window,fullscreen
				glfwMaximizeWindow(window);
				glfwSetWindowAttrib(window, GLFW_MAXIMIZED, GL_TRUE);
				glfwSetWindowAttrib(window, GLFW_DECORATED, GL_FALSE);
				glfwSetWindowMonitor(window, NULL, 0, 0, mode->width, mode->height, GLFW_DONT_CARE);
			}
		}
		else if (key == GLFW_KEY_T) {//enable/disable showing triangle faces
			sim.show_triangle = !sim.show_triangle;
		}
		else if (key == GLFW_KEY_R) {// reset
			sim.RESET = true;
			sim.SHOULD_RUN = true;
		}
		else if (key == GLFW_KEY_P) {//pause
			if (sim.RUNNING) { sim.pause(0); }
			else { sim.resume(); }
		}
		else if (key == GLFW_KEY_O) {//step
			if (!sim.RUNNING) { sim.resume(); }
			sim.setBreakpoint(sim.T + sim.NUM_QUEUED_KERNELS * sim.dt);
		}
		else if (key == GLFW_KEY_C) {
			switch (sim.joint_control.mode)
			{
			case JointControlMode::vel: // change to position control
				sim.joint_control.updateControlMode(JointControlMode::pos);
				printf("joint position control\n");
				break;
			case JointControlMode::pos: // change to velocity control
				sim.joint_control.updateControlMode(JointControlMode::vel);
				printf("joint speed control\n");
				break;
			}
		}
	}
	if (key == GLFW_KEY_W) { sim.camera_h_offset -= 0.05; }//camera moves closer
	else if (key == GLFW_KEY_S) { sim.camera_h_offset += 0.05; }//camera moves away
	else if (key == GLFW_KEY_A) { sim.camera_yaw -= 0.05; } //camera moves left
	else if (key == GLFW_KEY_D) { sim.camera_yaw += 0.05; }//camera moves right
	else if (key == GLFW_KEY_Q) { sim.camera_up_offset -= 0.05; } // camera moves down
	else if (key == GLFW_KEY_E) { sim.camera_up_offset += 0.05; }// camera moves up
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
