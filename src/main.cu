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

#include <GL/glew.h>// Include GLEW
#include <GLFW/glfw3.h>// Include GLFW
#include <glm/glm.hpp>// Include GLM
#include <glm/gtc/matrix_transform.hpp>

#define _USE_MATH_DEFINES
#include <math.h>

#include <omp.h>

#include <msgpack.hpp>

#include "vec.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = false)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) {
			char buffer[200];
			snprintf(buffer, sizeof(buffer), "GPUassert error in CUDA kernel: %s %s %d\n", cudaGetErrorString(code), file, line);
			std::string buffer_string = buffer;
			throw std::runtime_error(buffer_string);
		}
	}
}


struct MASS {
	double* m;
	Vec* pos;
	Vec* vel;
	Vec* acc;
	Vec* force;
	Vec* force_extern;
	Vec* color;
	bool* fixed;
	MASS() {}
	MASS(size_t size, bool onGpu = false) {
		if (onGpu) { // allocate on device GPU
			gpuErrchk(cudaMalloc((void**)&m, size * sizeof(double)));
			gpuErrchk(cudaMalloc((void**)&pos, size * sizeof(Vec)));
			gpuErrchk(cudaMalloc((void**)&vel, size * sizeof(Vec)));
			gpuErrchk(cudaMalloc((void**)&acc, size * sizeof(Vec)));
			gpuErrchk(cudaMalloc((void**)&force, size * sizeof(Vec)));
			gpuErrchk(cudaMalloc((void**)&force_extern, size * sizeof(Vec)));
			gpuErrchk(cudaMalloc((void**)&color, size * sizeof(Vec)));
			gpuErrchk(cudaMalloc((void**)&fixed, size * sizeof(bool)));
		}
		else { // allocate on host cpu
			gpuErrchk(cudaMallocHost((void**)&m, size * sizeof(double)));
			gpuErrchk(cudaMallocHost((void**)&pos, size * sizeof(Vec)));
			gpuErrchk(cudaMallocHost((void**)&vel, size * sizeof(Vec)));
			gpuErrchk(cudaMallocHost((void**)&acc, size * sizeof(Vec)));
			gpuErrchk(cudaMallocHost((void**)&force, size * sizeof(Vec)));
			gpuErrchk(cudaMallocHost((void**)&force_extern, size * sizeof(Vec)));
			gpuErrchk(cudaMallocHost((void**)&color, size * sizeof(Vec)));
			gpuErrchk(cudaMallocHost((void**)&fixed, size * sizeof(bool)));
		}
	}
};

struct SPRING {
	double* k; // spring constant (N/m)
	double* rest; // spring rest length (meters)
	double* damping; // damping on the masses.
	int* left; // index of left mass
	int* right; // index of right mass
	SPRING() {}
	SPRING(size_t size,bool onGpu = false) {
		if (onGpu) {
			gpuErrchk(cudaMalloc((void**)&k, size * sizeof(double)));
			gpuErrchk(cudaMalloc((void**)&rest, size * sizeof(double)));
			gpuErrchk(cudaMalloc((void**)&damping, size * sizeof(double)));
			gpuErrchk(cudaMalloc((void**)&left, size * sizeof(int)));
			gpuErrchk(cudaMalloc((void**)&right, size * sizeof(int)));
		}
		else { // allocate on host
			gpuErrchk(cudaMallocHost((void**)&k, size * sizeof(double)));
			gpuErrchk(cudaMallocHost((void**)&rest, size * sizeof(double)));
			gpuErrchk(cudaMallocHost((void**)&damping, size * sizeof(double)));
			gpuErrchk(cudaMallocHost((void**)&left, size * sizeof(int)));
			gpuErrchk(cudaMallocHost((void**)&right, size * sizeof(int)));
		}
	}
};

template<class T> // alias template for pinned allocator
using ThurstHostVec = std::vector<T, thrust::system::cuda::experimental::pinned_allocator<T>>;

struct Joint {
	std::vector<int> left;// the indices of the left points
	std::vector<int> right;// the indices of the right points
	std::vector<int> anchor;// the indices of the anchor points
	MSGPACK_DEFINE(left, right, anchor);
};
class Model {
public:
	std::vector<std::vector<double> > vertices;// the mass xyzs
	std::vector<std::vector<int> > edges;//the spring ids
	std::vector<int> idVertices;// the edge id of the vertices
	std::vector<int> idEdges;// the edge id of the springs
	std::vector<std::vector<double> > colors;// the mass xyzs
	std::vector<Joint> Joints;// the mass xyzs
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

#ifdef GRAPHICS
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

GLFWwindow* createGLFWWindow() {
	// Initialise GLFW
	if (!glfwInit()) { throw(std::runtime_error("Failed to initialize GLFW\n")); }
	//// MSAA: multisampling
	glfwWindowHint(GLFW_SAMPLES, GLFW_DONT_CARE); // #samples to use for multisampling. Zero disables multisampling.
	glEnable(GL_MULTISAMPLE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); // use GLSL 4.6
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // meke opengl forward compatible
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); //We don't want the old OpenGL
	glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
	glfwSwapInterval(false);// disable vsync
	// Open a window and create its OpenGL context
	GLFWwindow* window = glfwCreateWindow(1920, 1080, "CUDA Physics Simulation", NULL, NULL);
	if (window == NULL) {
		fprintf(stderr,
			"Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
		getchar();
		glfwTerminate();
		exit(1);
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
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
	return window;
}

#endif

class Simulation {
public:
	double dt = 0.001;
	double T = 0; //simulation time
	Vec global_acc = Vec(); // global acceleration

	// host
	MASS mass;
	SPRING spring;
	// device
	MASS d_mass;
	SPRING d_spring;

	// control
	bool RUNNING = false;
	bool STARTED = false;
	bool ENDED = false;
	// graphics
	int lineWidth = 1;
	int pointSize = 3;

	Simulation(size_t num_mass, size_t num_spring) {
		mass = MASS(num_mass,false);
		d_mass = MASS(num_mass, true);

		spring = SPRING(num_spring,false);
		d_spring = SPRING(num_spring, true);
	}


};

int main()
{
	auto start = std::chrono::steady_clock::now();
	Model bot("..\\src\\data.msgpack");

	int num_mass = bot.vertices.size();
	int num_spring = bot.edges.size();


	Simulation sim = Simulation(num_mass, num_spring);

	MASS& mass = sim.mass;
	SPRING& spring = sim.spring;

	//ThurstHostVec<double> h_mass_m(num_mass);
	//ThurstHostVec<Vec> h_mass_pos(num_mass);
	//ThurstHostVec<Vec> h_mass_vel(num_mass);
	//ThurstHostVec<Vec> h_mass_acc(num_mass);
	//ThurstHostVec<Vec> h_mass_force(num_mass);
	//ThurstHostVec<Vec> h_mass_force_extern(num_mass);
	//ThurstHostVec<Vec> h_mass_color(num_mass);
	//ThurstHostVec<bool> h_mass_fixed(num_mass);
	//ThurstHostVec<double> h_spring_k(num_spring);
	//ThurstHostVec<double> h_spring_rest(num_spring);
	//ThurstHostVec<double> h_spring_damping(num_spring);
	//ThurstHostVec<int> h_spring_left(num_spring);
	//ThurstHostVec<int> h_spring_right(num_spring);
	//h_mass_m.assign(num_mass, m);
	//h_spring_k.assign(num_spring, spring_constant);

	double dt = 0.001;
	double m = 0.01;// mass per vertex
	double spring_constant = 1e4;



#pragma omp simd
	for (size_t i = 0; i < num_mass; i++)
	{
		mass.pos[i] = bot.vertices[i];
		mass.color[i] = bot.colors[i];
		mass.m[i] = m; // mass [kg]
	}
#pragma omp simd
	for (size_t i = 0; i < num_spring; i++)
	{
		spring.left[i] = bot.edges[i][0];
		spring.right[i] = bot.edges[i][1];
		spring.k[i] = spring_constant; // spring constant
		spring.rest[i] = (mass.pos[spring.left[i]] - mass.pos[spring.right[i]]).norm();
	}

	auto end = std::chrono::steady_clock::now();
	std::cout << "Elapsed time:"
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
		<< " ms" << std::endl;

	return 0;
}