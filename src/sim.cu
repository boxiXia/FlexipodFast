/*modified from the orginal Titan simulation libaray:https://github.com/jacobaustin123/Titan
ref: J. Austin, R. Corrales-Fatou, S. Wyetzner, and H. Lipson, �Titan: A Parallel Asynchronous Library for Multi-Agent and Soft-Body Robotics using NVIDIA CUDA,� ICRA 2020, May 2020.
*/

#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#endif // !CUDA_API_PER_THREAD_DEFAULT_STREAM

#define GLM_FORCE_PURE
#include "sim.h"


__global__ void pbdSolveDist(
	const MASS mass,
	const SPRING spring,
	const double dt
) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < spring.num) {
		Vec2i e = spring.edge[i];
		Vec3d n = mass.pos[e.y] - mass.pos[e.x];// distance constraint direction
		double w1 = mass.inv_m[e.x];
		double w2 = mass.inv_m[e.y];
		double w = w1 + w2;

		double d = n.norm(); 
		double c = d - spring.rest[i]; // distance constraint magnitude
		n /= (d > 1e-13 ? d : 1e-13);// normalized to unit vector (direction),
		double alpha_bar = spring.compliance[i] / (dt * dt);
		double delta_lambda = -c / (w + alpha_bar);
		Vec3d p = delta_lambda * n;
		
		//mass.pos[e.x].atomicVecAdd(-p  * w1);
		//mass.pos[e.y].atomicVecAdd(p * w2);

		// velocity damping
		Vec3d dpn = n.dot(mass.vel[e.y] - mass.vel[e.x]) * fmin(spring.damping[i] * dt*dt, 1.0)/w * n;
		mass.pos[e.x].atomicVecAdd((-p + dpn) * w1);
		mass.pos[e.y].atomicVecAdd(( p - dpn) * w2);

//#ifdef ROTATION
//		if (spring.resetable[i]) {
//			//spring.rest[i] = length;//reset the spring rest length if this spring is restable
//			spring.rest[i] = spring.rest[i] * 0.9 + 0.1 * d;//reset the spring rest length if this spring is restable
//		}
//#endif // ROTATION
	}
}

__global__ void pbdSolveVel(
	const MASS mass,
	const SPRING spring,
	const double dt
) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < spring.num) {
		Vec2i e = spring.edge[i];
		Vec3d n = mass.pos[e.y] - mass.pos[e.x];// distance constraint direction
		double w1 = mass.inv_m[e.x];
		double w2 = mass.inv_m[e.y];
		double w = w1 + w2;
		double d = n.norm();
		n /= (d > 1e-13 ? d : 1e-13);// normalized to unit vector (direction),

		Vec3d dpn = n.dot(mass.vel[e.y] - mass.vel[e.x]) * fmin(spring.damping[i] * dt * dt, 1.0) / w * n;
		mass.pos[e.x].atomicVecAdd(dpn * w1);
		mass.pos[e.y].atomicVecAdd(- dpn * w2);

	}
}


__global__ void pbdSolveContact(
	const MASS mass,
	const CUDA_GLOBAL_CONSTRAINTS c,
	const Vec3d global_acc,
	const double dt
) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < mass.num) {
		if (mass.constrain) { //only apply to constrain set of masses
			//Vec3d _force(force);
			for (int j = 0; j < c.num_planes; j++) { // global constraints
				c.d_planes[j].solveDist(
					mass.force[i], mass.pos[i], mass.pos_prev[i], mass.vel[i],dt); // todo fix this 
			}
			//for (int j = 0; j < c.num_balls; j++) {
			//	c.d_balls[j].applyForce(force, pos, vel);
			//}
			//mass.force_constraint[i] = force - _force;
		}
		mass.vel[i] = (mass.pos[i] - mass.pos_prev[i]) / dt;

		// moved from the start of the loop
		mass.pos_prev[i] = mass.pos[i];
		mass.vel[i] += dt * (mass.force_extern[i] * mass.inv_m[i] + global_acc);
		mass.pos[i] += dt * mass.vel[i];

	}
}




__global__ void updateSpring(
	const MASS mass,
	const SPRING spring
) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < spring.num) {
		Vec2i e = spring.edge[i];
		Vec3d s_vec = mass.pos[e.y] - mass.pos[e.x];// the vector from left to right
		double length = s_vec.norm(); // current spring length

		s_vec /= (length > 1e-13 ? length : 1e-13);// normalized to unit vector (direction), check instablility for small length

		Vec3d force = 1.0 / spring.compliance[i] * (spring.rest[i] - length) * s_vec; // normal spring force
		force += s_vec.dot(mass.vel[e.x] - mass.vel[e.y]) * spring.damping[i] * s_vec;// damping

		mass.force[e.y].atomicVecAdd(force); // need atomics here
		mass.force[e.x].atomicVecAdd(-force); // removed condition on fixed
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
		s_vec /= (length > 1e-13 ? length : 1e-13);// normalized to unit vector (direction), check instablility for small length

		Vec3d force = 1.0 / spring.compliance[i] * (spring.rest[i] - length) * s_vec; // normal spring force
		force += s_vec.dot(mass.vel[e.x] - mass.vel[e.y]) * spring.damping[i] * s_vec;// damping

		mass.force[e.y].atomicVecAdd(force); // need atomics here
		mass.force[e.x].atomicVecAdd(-force); // removed condition on fixed

#ifdef ROTATION
		if (spring.resetable[i]) {
			//spring.rest[i] = length;//reset the spring rest length if this spring is restable
			spring.rest[i] = spring.rest[i] * 0.9 + 0.1 * length;//reset the spring rest length if this spring is restable
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
			Vec3d pos = mass.pos[i];
			Vec3d vel = mass.vel[i];
			mass.pos_prev[i] = pos;

			Vec3d force = mass.force[i];
			force += mass.force_extern[i];// add spring force and external force [N]
			force += global_acc / mass.inv_m[i];// add global accleration

			if (mass.constrain) { //only apply to constrain set of masses
				Vec3d _force(force);
				for (int j = 0; j < c.num_planes; j++) { // global constraints
					c.d_planes[j].applyForce(force, pos, vel); // todo fix this 
				}
				for (int j = 0; j < c.num_balls; j++) {
					c.d_balls[j].applyForce(force, pos, vel);
				}
				mass.force_constraint[i] = force - _force;
			}

			// euler integration
			force *= mass.inv_m[i];// force is now acceleration
			//force += global_acc;// add global accleration
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
__global__ void updateJoint(Vec3d* __restrict__ mass_pos,bool* __restrict__ mass_fixed, const JOINT joint) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < joint.vert_num) {
		int vert_id = joint.vert_id[i];
		if (!mass_fixed[vert_id]) {
			int vert_joint_id = joint.vert_joint_id[i];
			Vec2i anchor_edge = joint.anchor[vert_joint_id]; // mass id of the achor edge point
			mass_pos[vert_id] = AxisAngleRotaion(
				mass_pos[anchor_edge.x],
				mass_pos[anchor_edge.y], mass_pos[vert_id],
				joint.theta[vert_joint_id] * joint.vert_dir[i]);
		}
	}
}


__global__ void updateJointPos(Vec3d* __restrict__ mass_pos, double* __restrict__ spring_rest, const JOINT joint) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < joint.num) {
		Vec2i anchor_edge = joint.anchor[i];
		Vec3d rotation_axis = (mass_pos[anchor_edge.y] - mass_pos[anchor_edge.x]).normalize();
		//Vec3d x_left = mass.pos[joint.anchors.left_coord[i] + 1] - mass.pos[joint.anchors.left_coord[i]];//oxyz
		//Vec3d x_right = mass.pos[joint.anchors.right_coord[i] + 1] - mass.pos[joint.anchors.right_coord[i]];//oxyz
		//pos[i] = signedAngleBetween(x_left, x_right, rotation_axis); //joint angle in [-pi,pi]
		Vec3d y_left = mass_pos[joint.left_coord[i] + 2] - mass_pos[joint.left_coord[i]];//oxyz
		Vec3d y_right = mass_pos[joint.right_coord[i] + 2] - mass_pos[joint.right_coord[i]];//oxyz
		joint.pos[i] = signedAngleBetween(y_left, y_right, rotation_axis); //joint angle in [-pi,pi]
	}
}


__global__ void updateJointSpring(Vec3d* __restrict__ mass_pos, double* __restrict__ spring_rest, const JOINT joint) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < joint.num) {
		double delta = joint.pos_desired[i] - joint.pos[i];
		clampPeroidicInplace(delta, -M_PI, M_PI);
		joint.pos[i] += 0.02 * delta;
	}
	if (i < joint.edge_num) {
		Vec3d c = joint.edge_c[i]; // joint edge constant
		int edge_id = joint.edge_id[i];
		double joint_pos = joint.pos[joint.edge_joint_id[i]];
		//spring_rest[edge_id] = sqrt(c.x + c.y * cos(joint_pos + c.z));
		spring_rest[edge_id] = 0.99*spring_rest[edge_id]+0.01*sqrt(c.x + c.y * cos(joint_pos + c.z));
	}
}

#endif // ROTATION


#ifdef GRAPHICS

// update vertex positions
__global__ void updateVertices(VERTEX_DATA* __restrict__ gl_ptr, const Vec3d* __restrict__  pos, const int num_mass) {
	// https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
	// https://devblogs.nvidia.com/cuda-pro-tip-optimize-pointer-aliasing/
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_mass; i += blockDim.x * gridDim.x) {
		Vec3d pos_i = pos[i];

		auto& v = gl_ptr[i]; // vertex_i
		// update positions
		v.pos.x = (GLfloat)pos_i.x;
		v.pos.y = (GLfloat)pos_i.y;
		v.pos.z = (GLfloat)pos_i.z;
		// zero vertex normals, must run before update normals
		v.normal.x = 0.f;
		v.normal.y = 0.f;
		v.normal.z = 0.f;
	}
}

// update color rgb
__global__ void updateColors(VERTEX_DATA* __restrict__ gl_ptr, const Vec3d* __restrict__ color, const int num_mass) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_mass; i += blockDim.x * gridDim.x) {
		Vec3d color_i = color[i];
		auto& ptr_color_i = gl_ptr[i].color;
		ptr_color_i.x = (GLfloat)color_i.x;
		ptr_color_i.y = (GLfloat)color_i.y;
		ptr_color_i.z = (GLfloat)color_i.z;
	}
}

// update vertex normals from triangles, should run after updateVertices
__global__ void updateTriangleVertexNormal(
	VERTEX_DATA* __restrict__ gl_ptr,
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

		auto& n0 = gl_ptr[t.x].normal;
		atomicAdd(&n0.x, nx);
		atomicAdd(&n0.y, ny);
		atomicAdd(&n0.z, nz);

		auto& n1 = gl_ptr[t.y].normal;
		atomicAdd(&n1.x, nx);
		atomicAdd(&n1.y, ny);
		atomicAdd(&n1.z, nz);

		auto& n2 = gl_ptr[t.z].normal;
		atomicAdd(&n2.x, nx);
		atomicAdd(&n2.y, ny);
		atomicAdd(&n2.z, nz);
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

__host__ Simulation::Simulation(size_t num_mass, size_t num_spring, size_t num_joint, size_t num_triangle, int device) :
	device(device)
#ifdef UDP
	, udp_server(port_local, port_remote, ip_remote)// port_local,port_remote,ip_remote,num_joint
#endif //UDP
{
	gpuErrchk(cudaSetDevice(device)); // set cuda device
	mass = MASS(num_mass, true);// allocate host
	d_mass = MASS(num_mass, false);// allocate device
	spring = SPRING(num_spring, true);// allocate host
	d_spring = SPRING(num_spring, false);// / allocate device
	triangle = TRIANGLE(num_triangle, true);//allocate host
	d_triangle = TRIANGLE(num_triangle, false);//allocate device
	joint_control = JointControl(num_joint, true); // joint controller, must also call reset, see updatePhysics()

	for (int i = 0; i < NUM_CUDA_STREAM; ++i) { // lower i = higher priority
		cudaStreamCreateWithPriority(&stream[i], cudaStreamDefault, i);// create extra cuda stream: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution
	}

}

__host__ Simulation::~Simulation() {
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


__host__ void Simulation::start() {
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
	gpuErrchk(cudaSetDevice(device)); // set cuda device

	//cudaDeviceProp deviceProp;
	//cudaGetDeviceProperties(&deviceProp, 0);
	//if (deviceProp.cooperativeLaunch == 0) {
	//	printf("not supported");
	//	exit(-1);
	//}


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
					// send message every 1 ms or received new message (1000Hz)
					std::chrono::steady_clock::time_point ct_end = std::chrono::steady_clock::now();
					float diff = std::chrono::duration_cast<std::chrono::milliseconds>(ct_end - ct_begin).count();
					if (diff >= 1 || msg_received) {
						SHOULD_SEND_UDP = true;// send udp message
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

		for (int i = 0; i < joint.size(); i++) { // compute joint angles and angular velocity
			joint.theta[i] = NUM_UPDATE_PER_ROTATION * joint_control.cmd[i] * dt;// update joint speed
			joint.pos_desired[i] = joint_control.pos_desired[i];
		}
		d_joint.copyThetaFrom(joint, stream[CUDA_DYNAMICS_STREAM]);
		//d_joint.copyPosFrom(joint, stream[CUDA_DYNAMICS_STREAM]);
		cudaMemcpyAsync(d_joint.pos_desired, joint.pos_desired, joint.num * sizeof(double), cudaMemcpyDefault, stream[CUDA_DYNAMICS_STREAM]);


		// invoke cuda kernel for dynamics update

		if (USE_PBD) {
			for (int i = 0; i < NUM_QUEUED_KERNELS; i++) {
				updateJointSpring << <joint_edge_grid_size, joint_edge_block_size, 0, stream[CUDA_DYNAMICS_STREAM] >> > (d_mass.pos,d_spring.rest, d_joint);
				pbdSolveDist << <spring_grid_size, spring_block_size, 0, stream[CUDA_DYNAMICS_STREAM] >> > (d_mass, d_spring, dt);
				//updateJoint << <joint_grid_size, joint_block_size, 0, stream[CUDA_DYNAMICS_STREAM] >> > (d_mass.pos,d_mass.fixed, d_joint);
				pbdSolveContact << < mass_grid_size, mass_block_size, 0, stream[CUDA_DYNAMICS_STREAM] >> > (d_mass, d_constraints, global_acc, dt);
				//pbdSolveVel << <spring_grid_size, spring_block_size, 0, stream[CUDA_DYNAMICS_STREAM] >> > (d_mass, d_spring, dt);
			}
		}
		else{
			for (int i = 0; i < NUM_QUEUED_KERNELS; i++) {
#ifdef ROTATION
				if (k_rot % NUM_UPDATE_PER_ROTATION == 0) {
					k_rot = 0; // reset counter
					//updateJointSpring << <joint_edge_grid_size, joint_edge_block_size, 0, stream[CUDA_DYNAMICS_STREAM] >> > (d_mass.pos,d_spring.rest, d_joint);
					updateJoint << <joint_grid_size, joint_block_size, 0, stream[CUDA_DYNAMICS_STREAM] >> > (d_mass.pos,d_mass.fixed, d_joint);
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
		}


		T += NUM_QUEUED_KERNELS * dt;

		//mass.CopyPosVelAccFrom(d_mass, stream[CUDA_DYNAMICS_STREAM]);
		mass.CopyPosFrom(d_mass, stream[CUDA_DYNAMICS_STREAM]);
		mass.CopyConstraintForceFrom(d_mass, stream[CUDA_DYNAMICS_STREAM]); // copy force_constraint

		cudaStreamSynchronize(stream[CUDA_DYNAMICS_STREAM]);


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
		case UDP_HEADER::MOTOR_POS_COMMEND:
			joint_control.updateControlMode(JointControlMode::pos);
			for (int i = 0; i < joint_control.size(); i++) {//update joint speed from received udp packet
				joint_control.pos_desired[i] = udp_server.msg_rec.joint_value_desired[i];
			}
			if (!RUNNING) { SHOULD_RUN = true; }
			break;
		case UDP_HEADER::STEP_MOTOR_POS_COMMEND:
			joint_control.updateControlMode(JointControlMode::pos);
			for (int i = 0; i < joint_control.size(); i++) {//update joint speed from received udp packet
				joint_control.pos_desired[i] = udp_server.msg_rec.joint_value_desired[i];
			}
			if (!RUNNING) { SHOULD_RUN = true; }
			setBreakpoint(T + NUM_QUEUED_KERNELS * dt);
			break;
		case UDP_HEADER::MOTOR_SPEED_COMMEND:
			joint_control.updateControlMode(JointControlMode::vel);
			for (int i = 0; i < joint_control.size(); i++) {//update joint speed from received udp packet
				joint_control.vel_desired[i] = udp_server.msg_rec.joint_value_desired[i];
			}
			if (!RUNNING) { SHOULD_RUN = true; }
			break;
		case UDP_HEADER::STEP_MOTOR_SPEED_COMMEND:
			joint_control.updateControlMode(JointControlMode::vel);
			for (int i = 0; i < joint_control.size(); i++) {//update joint speed from received udp packet
				joint_control.vel_desired[i] = udp_server.msg_rec.joint_value_desired[i];
			}
			if (!RUNNING) { SHOULD_RUN = true; }
			setBreakpoint(T + NUM_QUEUED_KERNELS * dt);
			break;
		case UDP_HEADER::TERMINATE://close the program
			bpts.insert(BreakPoint(0, true));//SHOULD_END = true;
			break;
		case UDP_HEADER::RESET: // reset
			// set the reset flag to true, resetState() will be called 
			//RESET = true;// to restore mass/spring/joint state to the backedup state
			SHOULD_RUN = true;
			resetState();// restore the robot mass/spring/joint state to the backedup state
			break;
		case UDP_HEADER::PAUSE:
			bpts.insert(BreakPoint(0, false));//SHOULD_END = true;
			break;
		case UDP_HEADER::RESUME:
			SHOULD_RUN = true;
			break;
		default:
			break;
		}
		return true;
	}
	else { return false; }
}

void Simulation::updateUdpMessage() {

	gpuErrchk(cudaSetDevice(device)); // set cuda device

	auto _msg_send = udp_server.msg_send;//copy constuct
	double _T = -1; // time at last send

	while (!SHOULD_END) {
		if (SHOULD_SEND_UDP) {
			SHOULD_SEND_UDP = false;
			if (_T != T) {
				_T = T; // update time last send
				body.update(mass, id_oxyz_start, NUM_QUEUED_KERNELS * dt);
				_msg_send.emplace_front(
					DataSend(UDP_HEADER::ROBOT_STATE_REPORT, T, joint_control, body
#ifdef STRESS_TEST	
						, id_selected_edges, mass, spring
#endif //STRESS_TEST
					));

#ifdef MEASURE_CONSTRAINT
				Vec3d _fc;// constraint force
				
				for (int i = 0; i < mass.size(); i++)
				{
					if (mass.constrain[i]) {
						_fc += mass.force_constraint[i];
					}
				}

				fc_arr_x[fc_arr_idx] = (float)_fc.x;
				fc_arr_y[fc_arr_idx] = (float)_fc.y;
				fc_arr_z[fc_arr_idx] = (float)_fc.z;

				fc_arr_idx = (fc_arr_idx+1)%fc_arr_x.size();

				float f_norm = (float)_fc.norm();
				fc_max = std::max(fc_max, f_norm);

				force_constraint = _fc;
#endif //MEASURE_CONSTRAINT

			}

			if (UDP_INIT) {
				for (int i = 1; i < udp_num_obs * udp_step; i++) { // replicate up to udp_num_obs*udp_step times
					_msg_send.push_front(_msg_send.front());
					//printf("udp init # %d\n", udp_server.msg_send.size());
				}
				UDP_INIT = false;
			}
			while (_msg_send.size() > udp_num_obs * udp_step) {
				_msg_send.pop_back();
			}
			// sending..
			udp_server.msg_send.clear();// clearing first
			for (int i = 0; i < udp_num_obs; i++) { // replicate up to udp_num_obs times
				udp_server.msg_send.push_back(_msg_send[i * udp_step]);
				//printf("udp init # %d\n", udp_server.msg_send.size());
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




void Simulation::freeGPU() {
	d_balls.clear();
	d_balls.shrink_to_fit();

	d_planes.clear();
	d_planes.shrink_to_fit();
	printf("GPU freed\n");
}


// creates half-space ax + by + cz < d
void Simulation::createPlane(const Vec3d& abc, const double d, const double FRICTION_K, const double FRICTION_S,
	float square_size, float plane_radius) { // creates half-space ax + by + cz < d
	if (ENDED) { throw std::runtime_error("The simulation has ended. New objects cannot be created."); }
	ContactPlane* new_plane = new ContactPlane(abc, d, square_size, plane_radius);
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
		e_potential += -global_acc.dot(mass.pos[i]) / mass.inv_m[i];
		e_kinetic += 0.5 * mass.vel[i].SquaredSum() / mass.inv_m[i];
	}
	for (int i = 0; i < spring.num; i++)
	{
		e_potential += 0.5/spring.compliance[i] * pow((mass.pos[spring.left[i]] - mass.pos[spring.right[i]]).norm() - spring.rest[i], 2);
	}
	return e_potential + e_kinetic;
}
#endif // DEBUG_ENERGY



/* compute the block size (threads per block) and grid size （blocks per grid)
*/
int Simulation::computeGridSize(int block_size, int num) {
	int grid_size = (num - 1 + block_size) / block_size;// Round up according to array size 
	if (grid_size < 1 || (grid_size > MAX_BLOCKS))  // kernel has a hard limit on MAX_BLOCKS
		throw std::exception("computeGridSize():grid size excpetion!"); 
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
	joint_grid_size = computeGridSize(joint_block_size, joint.vert_num);

	// joint edge
	joint_edge_grid_size = computeGridSize(joint_edge_block_size, joint.edge_num);

#ifdef GRAPHICS
	//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &triangle_block_size, updateTriangleVertexNormal, 0, 0);
	triangle_grid_size = computeGridSize(triangle_block_size, triangle.size());

	//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &vertex_block_size, updateVertices, 0, 0);
	vertex_grid_size = computeGridSize(vertex_block_size, mass.size());
#endif //GRAPHICS

}


void Simulation::setAll() {//copy form cpu
	d_mass.copyFrom(mass, stream[CUDA_MEMORY_STREAM]);
	d_spring.copyFrom(spring, stream[CUDA_MEMORY_STREAM]);
	d_joint.copyFrom(joint, stream[CUDA_MEMORY_STREAM]);
#ifdef GRAPHICS
	d_triangle.copyFrom(triangle, stream[CUDA_MEMORY_STREAM]);
#endif //GRAPHICS
	//cudaDeviceSynchronize();
}

void Simulation::getAll() {//copy from gpu
	mass.copyFrom(d_mass, stream[CUDA_MEMORY_STREAM]); // mass
	spring.copyFrom(d_spring, stream[CUDA_MEMORY_STREAM]);// spring
	joint.copyFrom(d_joint, stream[CUDA_MEMORY_STREAM]);
#ifdef GRAPHICS
	triangle.copyFrom(d_triangle, stream[CUDA_MEMORY_STREAM]);// triangle
#endif //GRAPHICS
	//cudaDeviceSynchronize();
}


/*backup the robot mass/spring/joint state */
void Simulation::backupState() {
	backup_spring = SPRING(spring, true);
	backup_mass = MASS(mass, true);
	backup_joint = JOINT(joint, true);
}
/*restore the robot mass/spring/joint state to the backedup state *///TODO check if other variable needs resetting
void Simulation::resetState() {//TODO...fix bug
	cudaStreamSynchronize(stream[CUDA_DYNAMICS_STREAM]);
	d_mass.copyFrom(backup_mass, stream[CUDA_MEMORY_STREAM]);
	mass.copyFrom(backup_mass, stream[CUDA_MEMORY_STREAM_ALT]);

	d_spring.copyFrom(backup_spring, stream[CUDA_MEMORY_STREAM]);
	spring.copyFrom(backup_spring, stream[CUDA_MEMORY_STREAM_ALT]);

	d_joint.copyFrom(backup_joint, stream[CUDA_MEMORY_STREAM]);
	joint.copyFrom(backup_joint, stream[CUDA_MEMORY_STREAM_ALT]);

	joint_control.reset(backup_mass, backup_joint);
	//joint_control.update(backup_mass, backup_joint, dt);
	body.init(backup_mass, id_oxyz_start); // init body frame
	cudaStreamSynchronize(stream[CUDA_MEMORY_STREAM]);

#ifdef UDP
	UDP_INIT = true; // tell the udp thread to initiailze
	SHOULD_SEND_UDP = true; // tell the udp to send
#endif // UDP
	RESET = false; // set reset to false 
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


#ifdef GRAPHICS
void Simulation::updateGraphics() {
	gpuErrchk(cudaSetDevice(device)); // set cuda device

	createGLFWWindow(); // create a window with  width and height
	startupImgui(); // Setup Dear ImGui

	glGenVertexArrays(1, &VertexArrayID);//GLuint VertexArrayID;
	glBindVertexArray(VertexArrayID);

	//int glDeviceId;// todo:this output wrong number, maybe it is a cuda bug...
	//unsigned int glDeviceCount;
	//cudaGLGetDevices(&glDeviceCount, &glDeviceId, 1u, cudaGLDeviceListAll);
	//printf("openGL device: %u\n", glDeviceId);

	// get the directory of this program
	std::string program_dir = getProgramDir();
	// Create and compile our GLSL program from the shaders
	shader = Shader(
		program_dir + "\\shader\\shaderVertex.glsl",
		program_dir + "\\shader\\shaderFragment.glsl");
	//shader.use();

	//https://learnopengl.com/Advanced-Lighting/Shadows/Point-Shadows
	//https://github.com/JoeyDeVries/LearnOpenGL/tree/master/src/5.advanced_lighting/3.1.3.shadow_mapping
	simpleDepthShader = Shader(
		program_dir + "\\shader\\shadow_mapping_depth_vertex.glsl",
		program_dir + "\\shader\\shadow_mapping_depth_fragment.glsl");
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

	for (Constraint* c : constraints) { // generate buffers for constraint objects
		c->generateBuffers();
	}
	generateBuffers(); // generate buffers for all masses and springs

	//updateBuffers();//Todo might not need?
	cudaDeviceSynchronize();//sync before while loop

	//double T_previous_update = T;	

	// check for errors
	GLenum error = glGetError();
	if (error != GL_NO_ERROR)
		std::cerr << "OpenGL Error " << error << std::endl;

	auto dt_graphic = std::chrono::nanoseconds(int(1e9 / 60.0));
	auto  refresh_time = std::chrono::steady_clock::now();
	while (!SHOULD_END) {
		auto t_now = std::chrono::steady_clock::now();
		if (t_now >= refresh_time)
		{
			refresh_time = t_now + dt_graphic;

			// graphics update loop
			if (resize_buffers) {
				resizeBuffers(); // needs to be run from GPU thread
				updateBuffers(); // full update
			}
			else {
				updateVertexBuffers(); // partial update
			}
			//cudaDeviceSynchronize(); // synchronize before updating the springs and mass positions



			double speed_multiplier = 0.2;
			double pos_multiplier = 0.1;

			if (glfwGetKey(window, GLFW_KEY_UP)) {
				for (int i = 0; i < joint.size(); i++) {
					if (joint_control.mode == JointControlMode::vel) {
						joint_control.vel_desired[i] += i < 2 ? speed_multiplier : -speed_multiplier;
					}
					else if (joint_control.mode == JointControlMode::pos) {
						joint_control.pos_desired[i] += i < 2 ? pos_multiplier : -pos_multiplier;
					}
				}
			}
			else if (glfwGetKey(window, GLFW_KEY_DOWN)) {
				for (int i = 0; i < joint.size(); i++) {
					if (joint_control.mode == JointControlMode::vel) {
						joint_control.vel_desired[i] -= i < 2 ? speed_multiplier : -speed_multiplier;
					}
					else if (joint_control.mode == JointControlMode::pos) {
						joint_control.pos_desired[i] -= i < 2 ? pos_multiplier : -pos_multiplier;
					}
				}
			}
			if (glfwGetKey(window, GLFW_KEY_LEFT)) {
				for (int i = 0; i < joint.size(); i++) {
					if (joint_control.mode == JointControlMode::vel) {
						joint_control.vel_desired[i] -= speed_multiplier;
					}
					else if (joint_control.mode == JointControlMode::pos) {
						joint_control.pos_desired[i] -= pos_multiplier;
					}
				}
			}
			else if (glfwGetKey(window, GLFW_KEY_RIGHT)) {
				for (int i = 0; i < joint.size(); i++) {
					if (joint_control.mode == JointControlMode::vel) {
						joint_control.vel_desired[i] += speed_multiplier;
					}
					else if (joint_control.mode == JointControlMode::pos) {
						joint_control.pos_desired[i] += pos_multiplier;
					}
				}
			}
			else if (glfwGetKey(window, GLFW_KEY_0)) { // zero speed
				joint_control.reset(mass, joint);
			}

			// copy only the position
			//mass.CopyPosFrom(d_mass, stream[CUDA_MEMORY_STREAM]);
			//mass.CopyPosFrom(d_mass, id_oxyz_start,1,stream[CUDA_MEMORY_STREAM]);
			Vec3d com_pos = mass.pos[id_oxyz_start];// center of mass position (anchored body center)

			// Interpolate half way from original view to the new.
			float interp_factor = T < 2.0 ? 1 : 0.01; // 0.0 == original, 1.0 == new

			glm::vec3 target_pos = glm::vec3(com_pos.x, com_pos.y, com_pos.z);

			camera.follow(target_pos, interp_factor);

			computeMVP(true); // update MVP, also update camera matrix //todo

			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear screen


			// 1. render depth of scene to texture (from light's perspective)
			// --------------------------------------------------------------
			float near_plane = -5.0, far_plane = 50.f;
			//lightProjection = glm::perspective(glm::radians(45.0f), (GLfloat)SHADOW_WIDTH / (GLfloat)SHADOW_HEIGHT, near_plane, far_plane); // note that if you use a perspective projection matrix you'll have to change the light position as the current light position isn't enough to reflect the whole scene
			glm::mat4 lightProjection = glm::ortho(-2.0f, 2.0f, -2.0f, 2.0f, near_plane, far_plane);

			//lightView = glm::lookAt(light.direction, glm::vec3(0.0f), glm::vec3(0.0, 1.0, 0.0));
			glm::vec3 light_pos = glm::vec3(
				com_pos.x + light.direction.x,
				com_pos.y + light.direction.y,
				com_pos.z + light.direction.z);

			glm::mat4 lightView = glm::lookAt(
				light_pos, // camera position in World Space
				glm::vec3(com_pos.x, com_pos.y, com_pos.z),// look at position
				glm::vec3(0, 1.0, 0.0));  // camera up vector (set to 0,-1,0 to look upside-down)

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
			//shader.setMat4("MVP", MVP); // set MVP
			shader.setMat4("model", model_matrix); // set model matrix
			shader.setMat4("view", view_matrix);// set view matrix
			shader.setMat4("projection", projection_matrix); // set projection matrix
			shader.setVec3("viewPos", camera.pos); // set view position
			shader.setMat4("lightSpaceMatrix", lightSpaceMatrix);
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, depthMap);
			draw();
			/*--------------------------------------------------------------------*/

			// Swap buffers, render screen
			glfwPollEvents();

			runImgui();// run ImGui, show menu etc.


			// update new frame
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
		//std::this_thread::sleep_for(std::chrono::microseconds(100));
	}

	shutdownImgui(); // imgui Cleanup and shutdown 

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


#ifdef GRAPHICS

void Simulation::generateBuffers() {
	createVBO(&vbo_vertex, &cuda_resource_vertex, mass.size() * sizeof(VERTEX_DATA), cudaGraphicsRegisterFlagsNone, GL_ARRAY_BUFFER);//TODO CHANGE TO WRITE ONLY
	createVBO(&vbo_edge, &cuda_resource_edge, spring.size() * sizeof(uint2), cudaGraphicsRegisterFlagsNone, GL_ELEMENT_ARRAY_BUFFER);
	createVBO(&vbo_triangle, &cuda_resource_triangle, triangle.size() * sizeof(uint3), cudaGraphicsRegisterFlagsNone, GL_ELEMENT_ARRAY_BUFFER);
}

void Simulation::resizeBuffers() {
	////TODO>>>>>>
	resizeVBO(&vbo_vertex, mass.size() * sizeof(VERTEX_DATA), GL_ARRAY_BUFFER);//TODO CHANGE TO WRITE ONLY
	resizeVBO(&vbo_edge, spring.size() * sizeof(uint2), GL_ELEMENT_ARRAY_BUFFER);
	resizeVBO(&vbo_triangle, triangle.size() * sizeof(uint3), GL_ELEMENT_ARRAY_BUFFER);
	resize_buffers = false;
}
void Simulation::deleteBuffers() {
	deleteVBO(&vbo_vertex, cuda_resource_vertex, GL_ARRAY_BUFFER);
	deleteVBO(&vbo_edge, cuda_resource_edge, GL_ELEMENT_ARRAY_BUFFER);
	deleteVBO(&vbo_triangle, cuda_resource_triangle, GL_ELEMENT_ARRAY_BUFFER);
}

/*----------------------------GL buffer----------------------------------*/

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

/*-----------------------end of GL buffer------------------------------*/

void Simulation::draw() {

	// draw constraints
	for (Constraint* c : constraints) {
		c->draw();
	}

	// ref: https://stackoverflow.com/questions/16380005/opengl-3-4-glvertexattribpointer-stride-and-offset-miscalculation
	glBindBuffer(GL_ARRAY_BUFFER, this->vbo_vertex);
	//glCheckError(); // check opengl error code
	glVertexAttribPointer( // vertex position
		0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		sizeof(VERTEX_DATA),// stride
		(void*)0            // array buffer offset
	);
	glEnableVertexAttribArray(0);

	//glCheckError(); // check opengl error code
	glVertexAttribPointer( // vertex color
		1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
		3,                                // size
		GL_FLOAT,                         // type
		GL_FALSE,                         // normalized?
		sizeof(VERTEX_DATA),              // stride
		(GLvoid*)(sizeof(VERTEX_DATA::pos))   // array buffer offset
	);
	glEnableVertexAttribArray(1);

	glVertexAttribPointer( // vertex normal
		2,                                // attribute. No particular reason for 1, but must match the layout in the shader.
		3,                                // size
		GL_FLOAT,                         // type
		GL_FALSE,                         // normalized?
		sizeof(VERTEX_DATA),              // stride
		(GLvoid*)(sizeof(VERTEX_DATA::pos) + sizeof(VERTEX_DATA::color))   // array buffer offset
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
}

void Simulation::setViewport(const glm::vec3& camera_position, const glm::vec3& target_location, const glm::vec3& up_vector) {
	camera.pos = camera_position;
	camera.dir = camera_position;
	camera.up = up_vector;

	//// initialized the camera_horizontal reference direction
	//camera_href_dir = Vec3d(1, 0, 0);
	//if (abs(camera_href_dir.dot(camera_up)) > 0.9) { camera_href_dir = Vec3d(0, 1, 0); }
	//camera_href_dir = camera_href_dir.decompose(camera_up).normalize();

	if (STARTED) { computeMVP(); }
}
void Simulation::moveViewport(const glm::vec3& displacement) {
	camera.pos += displacement;
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
		// projection_matrix matrix : 70 deg Field of view_matrix, width:height ratio, display range : 0.01 unit <-> 100 units
		this->projection_matrix = glm::perspective(glm::radians(70.0f), (float)framebuffer_width / (float)framebuffer_height, 0.01f, 100.0f);
	}

	if (update_view) {
		model_matrix = glm::mat4(1.0f);// model matrix
		// Camera matrix
		this->view_matrix = camera.getViewMatrix();
	}
	if (is_resized || update_view) {
		this->MVP = model_matrix*projection_matrix * view_matrix; // Remember, matrix multiplication is the other way around
	}
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
		else if (key == GLFW_KEY_M) {
			sim.show_imgui = !sim.show_imgui;
		}
	}
	sim.camera.processKeyboard(key);
}


void Simulation::framebuffer_size_callback(GLFWwindow* window, int width, int height)
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
	glClearColor(clear_color.r, clear_color.g, clear_color.b, clear_color.a);

}

#endif


#ifdef GRAPHICS


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




#endif
