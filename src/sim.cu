/*modified from the orginal Titan simulation libaray:https://github.com/jacobaustin123/Titan
ref: J. Austin, R. Corrales-Fatou, S. Wyetzner, and H. Lipson, “Titan: A Parallel Asynchronous Library for Multi-Agent and Soft-Body Robotics using NVIDIA CUDA,” ICRA 2020, May 2020.
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

#define _USE_MATH_DEFINES
#include <math.h>

__global__ void SpringUpate(
	const MASS mass,
	const SPRING spring
) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < spring.num) {
		Vec2i e = spring.edge[i];
		Vec3d s_vec = mass.pos[e.y] - mass.pos[e.x];// the vector from left to right
		double length = s_vec.norm(); // current spring length

		s_vec /= (length > 1e-10 ? length : 1e-10);// normalized to unit vector (direction), check instablility for small length

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

#ifdef VERLET // verlet integration
__global__ void MassUpateStarter(
	const MASS mass,
	const CUDA_GLOBAL_CONSTRAINTS c,
	const Vec3d global_acc,
	const double dt) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < mass.num) {
		if (mass.fixed[i] == false) {
			Vec3d force = global_acc;
			double m = mass.m[i];
			force *= m; // force = d_mass.m[i] * global_acc;
			force += mass.force[i];
			force += mass.force_extern[i];// add external force [N]

			/*if (mass.constrain) */ {
				for (int j = 0; j < c.num_planes; j++) { // global constraints
					c.d_planes[j].applyForce(force, mass.pos[i], mass.vel[i]); // todo fix this 
				}
				for (int j = 0; j < c.num_balls; j++) {
					c.d_balls[j].applyForce(force, mass.pos[i]);
				}
			}
			// starter
			Vec3d acc = force / m;
			mass.acc[i] = acc;
			mass.prev_pos[i] = mass.pos[i];
			mass.vel[i] += acc * dt;
			mass.pos[i] = mass.prev_pos[i] + mass.vel[i] * dt;
			mass.force[i].setZero();
		}
	}
}
#endif

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
			Vec3d force = global_acc;
			force *= m; // force = d_mass.m[i] * global_acc;
			force += mass.force[i];
			force += mass.force_extern[i];// add external force [N]

			/*if (mass.constrain)*/ {
				for (int j = 0; j < c.num_planes; j++) { // global constraints
					c.d_planes[j].applyForce(force, pos, vel); // todo fix this 
				}
				for (int j = 0; j < c.num_balls; j++) {
					c.d_balls[j].applyForce(force, pos);
				}
			}

#ifdef VERLET // verlet integration
			// Störmer–Verlet:https://en.wikipedia.org/wiki/Verlet_integration
			Vec3d acc = force / m;
			Vec3d pos = 2 * mass.pos[i] - mass.prev_pos[i] + dt * dt * acc; // new pos
			mass.vel[i] = (pos - mass.pos[i]) / dt;
			mass.prev_pos[i] = mass.pos[i];
			mass.pos[i] = pos;
			mass.force[i].setZero();

#else // euler integration
			force /= m; // force is now acceleration
			vel += force * dt; // vel += acc*dt
			mass.acc[i] = force; // update acceleration
			mass.vel[i] = vel; // update velocity
			mass.pos[i] += vel * dt; // update position
			mass.force[i].setZero();
#endif // verlet
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
			Vec3d force = global_acc;
			force *= m; // force = d_mass.m[i] * global_acc;
			force += mass.force[i];
			force += mass.force_extern[i];// add external force [N]

			/*if (mass.constrain)*/ {
				for (int j = 0; j < c.num_planes; j++) { // global constraints
					c.d_planes[j].applyForce(force, pos, vel); // todo fix this 
				}
				for (int j = 0; j < c.num_balls; j++) {
					c.d_balls[j].applyForce(force, pos);
				}
			}

#ifdef VERLET // verlet integration
			// Störmer–Verlet:https://en.wikipedia.org/wiki/Verlet_integration
			Vec3d acc = force / m;
			Vec3d pos = 2 * mass.pos[i] - mass.prev_pos[i] + dt * dt * acc; // new pos
			mass.vel[i] = (pos - mass.pos[i]) / dt;
			mass.prev_pos[i] = mass.pos[i];
			mass.pos[i] = pos;
			mass.force[i].setZero();

#else // euler integration
			force /= m; // force is now acceleration
			vel += force * dt; // vel += acc*dt
			mass.acc[i] = force; // update acceleration
			mass.vel[i] = vel; // update velocity
			mass.pos[i] += vel * dt; // update position
			mass.force[i].setZero();
#endif // verlet
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


__global__ void rotateJoint(Vec3d* __restrict__ mass_pos, const JOINT joint) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < joint.points.num) {
		if (i < joint.anchors.num) {
			Vec2i e = joint.anchors.edge[i];
			joint.anchors.dir[i] = (mass_pos[e.y] - mass_pos[e.x]).normalize ();
		}
		__threadfence();
		__syncthreads();

		int anchor_id = joint.points.anchorId[i];
		int mass_id = joint.points.massId[i];

		mass_pos[mass_id] = AxisAngleRotaion(joint.anchors.dir[anchor_id], mass_pos[mass_id],
			joint.anchors.theta[anchor_id] * joint.points.dir[i], mass_pos[joint.anchors.edge[anchor_id].x]);

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

Simulation::Simulation(int num_mass, int num_spring) :Simulation() {
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
	jointBlocksPerGrid = computeBlocksPerGrid(MASS_THREADS_PER_BLOCK, d_joints.points.num);
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

	cudaMallocHost((void**)&joint_speeds_cmd, joints.size() * sizeof(double));//initialize joint speed (commended) array 
	cudaMallocHost((void**)&joint_speeds, joints.size() * sizeof(double));//initialize joint speed (measured) array 
	cudaMallocHost((void**)&joint_angles, joints.size() * sizeof(double));//initialize joint angle (measured) array 

	setAll();// copy mass and spring to gpu
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

	updateBuffers();//Todo might not need?
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

#ifdef VERLET
	SpringUpate << <springBlocksPerGrid, THREADS_PER_BLOCK >> > (d_mass, d_spring);
	MassUpateStarter << <massBlocksPerGrid, MASS_THREADS_PER_BLOCK >> > (d_mass, d_constraints, global_acc, dt);
	gpuErrchk(cudaPeekAtLastError());
#endif // VERLET

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
					auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
					printf("Elapsed time:%d ms for %.1f simulation time (%.2f)\n",
						duration, T, T / ((double)duration / 1000.));

					//for (Constraint* c : constraints) {
					//	delete c;
					//}
#ifdef GRAPHICS
					glDeleteBuffers(1, &vertexbuffer);
					glDeleteBuffers(1, &colorbuffer);
					glDeleteBuffers(1, &elementbuffer);
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
#ifdef ROTATION

			rotateJoint << <jointBlocksPerGrid, MASS_THREADS_PER_BLOCK>> > (d_mass.pos, d_joints);
			SpringUpate << <springBlocksPerGrid, THREADS_PER_BLOCK >> > (d_mass, d_spring);
			//massUpdateAndRotate << <massBlocksPerGrid + jointBlocksPerGrid, MASS_THREADS_PER_BLOCK >> > (d_mass, d_constraints, d_joints, global_acc, dt);

			MassUpate << <massBlocksPerGrid, MASS_THREADS_PER_BLOCK >> > (d_mass, d_constraints, global_acc, dt);

			//gpuErrchk(cudaPeekAtLastError());
#endif // ROTATION
			//cudaEventRecord(event_rotation, stream[0]);
			//cudaStreamWaitEvent(NULL, event_rotation, 0);
		}
		//cudaEventDestroy(event);//destroy the event, necessary to prevent memory leak
		//cudaEventDestroy(event_rotation);//destroy the event, necessary to prevent memory leak

		T += NUM_QUEUED_KERNELS * dt;

#ifdef UDP
		//if (fmod(T, 1. / 100.0) < NUM_QUEUED_KERNELS * dt) {
			mass.CopyPosVelAccFrom(d_mass, stream[NUM_CUDA_STREAM - 1]);
			cudaDeviceSynchronize();
//#pragma omp parallel for
			for (int i = 0; i < joints.anchors.num; i++)
			{
				Vec2i anchor_edge = joints.anchors.edge[i];
				Vec3d rotation_axis = (mass.pos[anchor_edge.y] - mass.pos[anchor_edge.x]).normalize();
				Vec3d x_left = mass.pos[joints.anchors.leftCoord[i] + 1] - mass.pos[joints.anchors.leftCoord[i]];//oxyz
				Vec3d x_right = mass.pos[joints.anchors.rightCoord[i] + 1] - mass.pos[joints.anchors.rightCoord[i]];//oxyz
				double angle = signedAngleBetween(x_left, x_right, rotation_axis); //joint angle in [-pi,pi]

				double delta_angle = angle - joint_angles[i];
				if (delta_angle > M_PI) {
					joint_speeds[i] = delta_angle - 2 * M_PI;
				}
				else if (delta_angle > -M_PI) {
					joint_speeds[i] = delta_angle;
				}
				else {
					joint_speeds[i] = delta_angle + 2 * M_PI;
				}
				joint_angles[i] = angle;
			}
			if (fmod(T, 1. / 10.0) < NUM_QUEUED_KERNELS * dt) {
				printf("% 6.1f: ",T); // time

				for (int i = 0; i < joints.anchors.num; i++) {
					//printf("%+ 6.1f   ", joint_angles[i] * M_1_PI*180.0);
					printf("%+ 7.1f ", joint_speeds[i] * M_1_PI * 30.0 / (NUM_QUEUED_KERNELS * dt)); // display joint speed in RPM
				}

				Vec3d body_com = mass.pos[id_oxyz_start];//body center of mass
				printf("(%+ 6.2f %+ 6.2f %+ 6.2f) ", body_com.x, body_com.y, body_com.z);

				printf("\r\r");
			}

			//printf("\n");
		//}
#endif

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


			//// https://en.wikipedia.org/wiki/Slerp
			////mass.pos[id_oxyz_start].print();
			//double t_lerp = 0.01;
			//Vec3d camera_dir_new = (mass.pos[id_oxyz_start] - camera_pos).normalize();
			///*camera_dir = (1 - t_lerp)* camera_dir + t_lerp* camera_dir_new;*/ //linear interpolation from camera_dir to camera_dir_new by factor t_lerp
			//// spherical linear interpolation from camera_dir to camera_dir_new by factor t_lerp
			//camera_dir = slerp(camera_dir, camera_dir_new, t_lerp);
			//camera_dir.normalize();


			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear screen



			if (glfwGetKey(window, GLFW_KEY_W)) {
				camera_pos += 0.02 * (camera_dir - camera_dir.dot(camera_up) * camera_up);
			}
			else if (glfwGetKey(window, GLFW_KEY_S)) {
				camera_pos -= 0.02 * (camera_dir - camera_dir.dot(camera_up) * camera_up);
			}

			if (glfwGetKey(window, GLFW_KEY_A)) {
				camera_dir = AxisAngleRotaion(camera_up, camera_dir, 0.02, Vec3d());
			}
			else if (glfwGetKey(window, GLFW_KEY_D)) {
				camera_dir = AxisAngleRotaion(camera_up, camera_dir, -0.02, Vec3d());
			}
			else if (glfwGetKey(window, GLFW_KEY_Q)) {
				camera_pos.z -= 0.02;
			}
			else if (glfwGetKey(window, GLFW_KEY_E)) {
				camera_pos.z += 0.02;
			}

			double speed_multiplier = 0.000005;
			//double speed_multiplier = 0.0001;
			bool speed_updated = false;
			if (glfwGetKey(window, GLFW_KEY_UP)) {
				if (joint_speeds_cmd[0] < max_joint_speed) { joint_speeds_cmd[0] += speed_multiplier; }
				if (joint_speeds_cmd[1] < max_joint_speed) { joint_speeds_cmd[1] += speed_multiplier; }
				if (joint_speeds_cmd[2] > -max_joint_speed) { joint_speeds_cmd[2] -= speed_multiplier; }
				if (joint_speeds_cmd[3] > -max_joint_speed) { joint_speeds_cmd[3] -= speed_multiplier; }
				speed_updated = true;
			}
			else if (glfwGetKey(window, GLFW_KEY_DOWN)) {
				if (joint_speeds_cmd[0] > -max_joint_speed) { joint_speeds_cmd[0] -= speed_multiplier; }
				if (joint_speeds_cmd[1] > -max_joint_speed) { joint_speeds_cmd[1] -= speed_multiplier; }
				if (joint_speeds_cmd[2] < max_joint_speed) { joint_speeds_cmd[2] += speed_multiplier; }
				if (joint_speeds_cmd[3] < max_joint_speed) { joint_speeds_cmd[3] += speed_multiplier; }
				speed_updated = true;
			}
			if (glfwGetKey(window, GLFW_KEY_LEFT)) {
				for (int i = 0; i < joints.size(); i++)
				{
					if (joint_speeds_cmd[i] > -max_joint_speed) { joint_speeds_cmd[i] -= speed_multiplier; }
				}
				speed_updated = true;
			}
			else if (glfwGetKey(window, GLFW_KEY_RIGHT)) {
				for (int i = 0; i < joints.size(); i++)
				{
					if (joint_speeds_cmd[i] < max_joint_speed) { joint_speeds_cmd[i] += speed_multiplier; }
				}
				speed_updated = true;
			}
			else if (glfwGetKey(window, GLFW_KEY_0)) {
				for (int i = 0; i < joints.size(); i++)
				{
					joint_speeds_cmd[i] = 0.;
				}
				speed_updated = true;
			}
			if (speed_updated) {// update joint speed
				for (int k = 0; k < joints.size(); k++) {
					joints.anchors.theta[k] = NUM_UPDATE_PER_ROTATION * joint_speeds_cmd[k];
				}
				d_joints.anchors.copyThetaFrom(joints.anchors, stream[0]);
			}


			computeMVP(true); // update MVP, also update camera matrix //todo

			glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);// update transformation "MVP" uniform


			for (Constraint* c : constraints) {
				c->draw();
			}

			//updateBuffers();
			updateVertexBuffers();

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


void Simulation::resume() {
	if (ENDED) { throw std::runtime_error("Simulation has ended. Cannot resume simulation."); }
	if (!STARTED) { throw std::runtime_error("Simulation has not started. Cannot resume before calling sim.start()."); }
	if (mass.num == 0) { throw std::runtime_error("No masses have been added. Add masses before simulation starts."); }
	updateCudaParameters();
	cudaDeviceSynchronize();
	RUNNING = true;
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
	}
	if (!FREED) {
		freeGPU();
	}
}

void Simulation::waitForEvent() {
	if (ENDED && !FREED) { throw std::runtime_error("Simulation has ended. can't call control functions."); }
	while (RUNNING) {
		std::this_thread::sleep_for(std::chrono::nanoseconds(100));
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
		// Projection matrix : 60° Field of View, 4:3 ratio, display range : 0.01 unit <-> 100 units
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

inline void Simulation::generateBuffers() {

	glGenBuffers(1, &colorbuffer);//bind color buffer
	glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
	glBufferData(GL_ARRAY_BUFFER, 3 * mass.num * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
	cudaGLRegisterBufferObject(colorbuffer);

	glGenBuffers(1, &elementbuffer);//bind element buffer
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, 2 * spring.num * sizeof(GLuint), NULL, GL_DYNAMIC_DRAW); // second argument is number of bytes
	cudaGLRegisterBufferObject(elementbuffer);

	glGenBuffers(1, &vertexbuffer);// bind vertex buffer
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, 3 * mass.num * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
	cudaGLRegisterBufferObject(vertexbuffer);

	//Todo: maybe unbind buffer? see updateBuffers()
}

inline void Simulation::resizeBuffers() {
	//    std::cout << "resizing buffers (" << masses.size() << " masses, " << springs.size() << " springs)." << std::endl;
	//    std::cout << "resizing buffers (" << d_masses.size() << " device masses, " << d_springs.size() << " device springs)." << std::endl;
		//cudaGLUnmapBufferObject(colorbuffer);//refer to updateBuffers()
	cudaGLUnregisterBufferObject(this->colorbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, this->colorbuffer);
	glBufferData(GL_ARRAY_BUFFER, 3 * mass.num * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
	cudaGLRegisterBufferObject(this->colorbuffer);

	cudaGLUnregisterBufferObject(this->elementbuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->elementbuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, 2 * spring.num * sizeof(GLuint), NULL, GL_DYNAMIC_DRAW); // second argument is number of bytes
	cudaGLRegisterBufferObject(this->elementbuffer);

	cudaGLUnregisterBufferObject(this->vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, 3 * mass.num * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
	cudaGLRegisterBufferObject(this->vertexbuffer);

	resize_buffers = false;
}

__global__ void updateVertices(float* __restrict__ gl_ptr, const Vec3d* __restrict__  pos, const int num_mass) {
	// https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
	// https://devblogs.nvidia.com/cuda-pro-tip-optimize-pointer-aliasing/
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_mass; i += blockDim.x * gridDim.x) {
		Vec3d pos_i = pos[i];
		gl_ptr[3 * i] = (float)pos_i.x;
		gl_ptr[3 * i + 1] = (float)pos_i.y;
		gl_ptr[3 * i + 2] = (float)pos_i.z;
	}
}

__global__ void updateIndices(unsigned int* __restrict__ gl_ptr, const Vec2i* __restrict__ edge, const int num_spring) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_spring; i += blockDim.x * gridDim.x) {
		Vec2i e = edge[i];

		gl_ptr[2 * i] = (unsigned int)e.x; // todo check if this is needed
		gl_ptr[2 * i + 1] = (unsigned int)e.y;
	}
}

__global__ void updateColors(float* __restrict__ gl_ptr, const Vec3d* __restrict__ color, const int num_mass) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_mass; i += blockDim.x * gridDim.x) {
		Vec3d color_i = color[i];
		gl_ptr[3 * i] = (float)color_i.x;
		gl_ptr[3 * i + 1] = (float)color_i.y;
		gl_ptr[3 * i + 2] = (float)color_i.z;
	}
}

void Simulation::updateBuffers() { // todo: check the kernel call
	{
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		void* vertexPointer;
		cudaGLMapBufferObject(&vertexPointer, vertexbuffer);
		updateVertices << <massBlocksPerGrid, MASS_THREADS_PER_BLOCK, 0, stream[0] >> > ((float*)vertexPointer, d_mass.pos, mass.num);
		cudaGLUnmapBufferObject(vertexbuffer);
	}
	if (update_colors) {
		glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
		void* colorPointer; // if no masses, springs, or colors are changed/deleted, this can be start only once
		cudaGLMapBufferObject(&colorPointer, colorbuffer);
		updateColors << <massBlocksPerGrid, MASS_THREADS_PER_BLOCK, 0, stream[1] >> > ((float*)colorPointer, d_mass.color, mass.num);
		cudaGLUnmapBufferObject(colorbuffer);
		update_colors = false;
	}
	if (update_indices) {
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);
		void* indexPointer; // if no masses or springs are deleted, this can be start only once
		cudaGLMapBufferObject(&indexPointer, elementbuffer);
		updateIndices << <springBlocksPerGrid, THREADS_PER_BLOCK, 0, stream[2] >> > ((unsigned int*)indexPointer, d_spring.edge, spring.num);
		cudaGLUnmapBufferObject(elementbuffer);
		update_indices = false;
	}
}

void Simulation::updateVertexBuffers() {

	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	void* vertexPointer;
	cudaGLMapBufferObject(&vertexPointer, vertexbuffer);
	updateVertices << <massBlocksPerGrid, MASS_THREADS_PER_BLOCK, 0, stream[0] >> > ((float*)vertexPointer, d_mass.pos, mass.num);
	cudaGLUnmapBufferObject(vertexbuffer);
}

inline void Simulation::draw() {
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, this->vertexbuffer);
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
	glBindBuffer(GL_ARRAY_BUFFER, this->colorbuffer);
	glVertexAttribPointer(
		1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
		3,                                // size
		GL_FLOAT,                         // type
		GL_FALSE,                         // normalized?
		0,                                // stride
		(void*)0                          // array buffer offset
	);

	glDrawArrays(GL_POINTS, 0, mass.num); // 3 indices starting at 0 -> 1 triangle
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
