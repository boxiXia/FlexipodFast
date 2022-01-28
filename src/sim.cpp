#include "sim.h"

#ifdef GRAPHICS
#include "imgui.h" // imgui
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h" // implot
#include <cfloat>
#endif



#ifdef GRAPHICS
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
#endif

Model::Model(const std::string& file_path, bool versbose) {
	// get the msgpack robot model
	// Deserialize the serialized data
	std::ifstream ifs(file_path, std::ifstream::in | std::ifstream::binary);
	std::stringstream buffer;
	buffer << ifs.rdbuf();
	msgpack::unpacked upd;//unpacked data
	msgpack::unpack(upd, buffer.str().data(), buffer.str().size());
	//    std::cout << upd.get() << std::endl;
	*this = (upd.get().as<Model>());
	if (versbose) {
		printf("Loaded %s\n", file_path.c_str());
		printf("radius_poisson=%.3e [m] \n", radius_poisson);
		printf("#vertices=%d, #edges=%d, #triangles = %d, #joints = %d\n",
			(int)vertices.size(), (int)edges.size(), (int)triangles.size(), (int)joints.size());

	}
}

#ifdef GRAPHICS

/*--------------------------------- ImGui ----------------------------------------*/

// Helper to display a little (?) mark which shows a tooltip when hovered.
// In your own code you may want to display an actual icon if you are using a merged icon fonts (see docs/FONTS.md)
// ref: https://github.com/ocornut/imgui/blob/master/imgui_demo.cpp
static void HelpMarker(const char* title, const char* help)
{
	ImGui::Text(title);
	if (ImGui::IsItemHovered())
	{
		ImGui::BeginTooltip();
		ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
		ImGui::TextUnformatted(help);
		ImGui::PopTextWrapPos();
		ImGui::EndTooltip();
	}
}

/*Setup Dear ImGui*/
void Simulation::startupImgui() {
	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	ImPlot::CreateContext();

	ImGuiIO& io = ImGui::GetIO(); (void)io;
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	//ImGui::StyleColorsClassic();

	// Setup Platform/Renderer backends
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	//const char* glsl_version = "#version 460"; //TODO change this in header
	std::ostringstream glsl_version;
	glsl_version << "#version " << contex_version_major << contex_version_minor << "0";
	ImGui_ImplOpenGL3_Init(glsl_version.str().c_str());

	//scale for high dpi 
	// https://doc.magnum.graphics/magnum/classMagnum_1_1ImGuiIntegration_1_1Context.html#ImGuiIntegration-Context-dpi
	auto monitor = glfwGetPrimaryMonitor();
	//const GLFWvidmode* mode = glfwGetVideoMode(monitor);
	//float xscale=2, yscale=2;
	float xscale, yscale;
	glfwGetMonitorContentScale(monitor, &xscale, &yscale);
	//std::cout << xscale << "," << yscale;
	std::string font_path = (getProgramDir() + "\\Cousine-Regular.ttf");

	io.Fonts->AddFontFromFileTTF(font_path.c_str(), 16.0f * xscale);
	auto& style = ImGui::GetStyle(); // set style
	style.ScaleAllSizes(xscale);
	style.FramePadding.y /= 2.0; // reduce vertical padding
}


/*run Imgui, processing inputs*/
void Simulation::runImgui() {

	// for measuring simulation speed
	static auto t_prev = std::chrono::steady_clock::now();
	static auto t_sim_prev = T;

	static double gravity_max = 10;
	static double gravity_min = -10;
	static double sim_speed = 1; // 

	static int counter_rec = 0;
	static float rec_fps = 0;

	if (show_imgui) {// show imgui window

		// Start the Dear ImGui frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();


		//// ref: https://github.com/ocornut/imgui/blob/master/imgui_demo.cpp
		//static bool show_demo_window = true;
		//if (ImGui::Button("show_demo")) { show_demo_window = true; }
		//if (show_demo_window) {
		//	ImGui::ShowDemoWindow(&show_demo_window);
		//	ImGui::ShowMetricsWindow(&show_demo_window);
		//	ImGui::ShowStyleEditor();
		//	ImPlot::ShowDemoWindow(&show_demo_window);
		//}

		// 
		// measure simulation speed
		auto t = std::chrono::steady_clock::now();
		float duration = (float)std::chrono::duration_cast<std::chrono::milliseconds>(t - t_prev).count() / 1000.;//[seconds]
		if (duration > 0.3) {
			float sim_duration = T - t_sim_prev;
			sim_speed = sim_duration / duration;
#ifdef UDP			
			rec_fps = (float(udp_server.counter_rec - counter_rec)) / sim_duration; // frame per simulation seconds
			counter_rec = udp_server.counter_rec;
#endif // UDP
			t_sim_prev = T;
			t_prev = t;
		}
		ImGui::Begin("Debug console", &show_imgui);

		if (ImGui::Button("Reset")) { RESET = true; SHOULD_RUN = true; }// reset state
		ImGui::SameLine();

		if (RUNNING) { if (ImGui::Button("Pause ")) { pause(0); } } // pause
		else if (ImGui::Button("Resume")) { resume(); }// resume
		ImGui::SameLine();

		// simulation time | simulation speed | rendering FPS
		ImGui::Text("%.2f s | % 5.2f X | %.1f FPS", T, sim_speed, ImGui::GetIO().Framerate);
#ifdef UDP	
		ImGui::Text("UDP rec %.2f FPSS", rec_fps);
#endif // UDP

		// physics
		if (joint_control.size() > 0 && ImGui::CollapsingHeader("physics")) {

			static double dt_min = 1e-7;
			static double dt_max = 1e-3;
			ImGui::DragScalar("dt", ImGuiDataType_Double, &dt, 1e-7, &dt_min, &dt_max, "%5.3e");
			ImGui::DragScalarN("gravity", ImGuiDataType_Double, &global_acc, 3, 0.1, &gravity_min, &gravity_max, "%.2f");
		}

		// joint control
		if (joint_control.size() > 0 && ImGui::CollapsingHeader("Joint control")) {
			if (ImGui::BeginTable("split", 5, ImGuiTableFlags_Resizable | ImGuiTableFlags_NoBordersInBodyUntilResize)){
				// We could also set ImGuiTableFlags_SizingFixedFit on the table and all columns will default to ImGuiTableColumnFlags_WidthFixed.
				ImGui::TableSetupColumn("id", ImGuiTableColumnFlags_WidthFixed); // Default to 100.0f
				ImGui::TableSetupColumn("x", ImGuiTableColumnFlags_WidthFixed); // Default to 200.0f
				ImGui::TableSetupColumn("x_d", ImGuiTableColumnFlags_WidthFixed,150.f);       // Default to auto
				ImGui::TableSetupColumn("v_d", ImGuiTableColumnFlags_WidthFixed, 150.f);       // Default to auto
				ImGui::TableSetupColumn("torque", ImGuiTableColumnFlags_WidthFixed);       // Default to auto

				ImGui::TableHeadersRow();

				

				for (int i = 0; i < joint_control.size(); i++) {
					if (i == 0) {
						for (int col = 0; col < 5; col++) {
							ImGui::TableSetColumnIndex(col);
							ImGui::PushItemWidth(-FLT_MIN); // Right-aligned
						}
					}
					ImGui::TableNextRow();

					ImGui::TableSetColumnIndex(0);
					ImGui::Text("%2d", i);

					ImGui::TableSetColumnIndex(1);
					ImGui::Text("%+6.3f", joint_control.pos[i]);

					ImGui::TableSetColumnIndex(2);
					char label_jpd[10];//joint_pos_des
					sprintf(label_jpd, "#jpd_%d", i);
					ImGui::DragScalar(label_jpd, ImGuiDataType_Double, &(joint_control.pos_desired[i]), 0.001f, NULL, NULL, "%6.3f");

					ImGui::TableSetColumnIndex(3);
					char label_jvd[10];//joint_pos_des
					sprintf(label_jvd, "#jvd_%d", i);//joint_vel_des
					ImGui::DragScalar(label_jvd, ImGuiDataType_Double, &(joint_control.vel_desired[i]), 0.005f, NULL, NULL, "%6.3f");

					ImGui::TableSetColumnIndex(4);
					ImGui::Text("%+6.3f", joint.torque[i]);
				}
				ImGui::EndTable();
			}
			
			// ref: https://github.com/ocornut/imgui/blob/838c16533d3a76b83f0ca73045010d463b73addf/imgui_demo.cpp#L687
			const char* elem_name = (joint_control.mode == JointControlMode::vel) ? "vel" : "pos";
			ImGui::SliderInt("control mode", &((int&)joint_control.mode), 0, 1, elem_name);
		
		}

		if (ImGui::CollapsingHeader("Info")) {
			char info[250];
			int n = body.print(info,250);
			ImGui::Text(info);
		}
#ifdef MEASURE_CONSTRAINT
		if (ImGui::CollapsingHeader("Constraint")) {
			static std::vector<float> arr = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60 };
			static int values_offset = 0;
			ImGui::Text("fc: %+6.1f %+6.1f %+6.1f N", force_constraint.x, force_constraint.y, force_constraint.z);
			ImGui::Text("fc_max: %+6.1f N", fc_max);
			if (ImGui::TreeNode("Per body constraint [N]##2")) {
				char info_str[1000];
				int n_char = 0;
				for (int i = 0; i < body_constraint_force.size(); i++)
				{
					auto& cfi = body_constraint_force[i];
					n_char += snprintf(info_str + n_char, 300, "%3d %+7.1f %+7.1f %+7.1f\n", i, cfi.x, cfi.y, cfi.z);
				}
				ImGui::Text(info_str);
				ImGui::TreePop();
			}

			double t_scale = NUM_QUEUED_KERNELS * dt;
			//ImPlot::SetupAxisLimits(ImAxis_X1,0, std::max(fc_arr.num,1) * t_scale, ImGuiCond_Always);
			int t_count = fc_arr.num;
			if (ImPlot::BeginPlot("constraint force [N]##5469", NULL, NULL, ImVec2(-1, 0),
				ImPlotFlags_None, ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_None)) {
				ImPlot::PlotLine("fx", (double*)fc_arr.data, t_count, t_scale, 0, fc_arr.idx, sizeof(Vec3d));
				ImPlot::PlotLine("fy", (double*)fc_arr.data + 1, t_count, t_scale, 0, fc_arr.idx, sizeof(Vec3d));
				ImPlot::PlotLine("fz", (double*)fc_arr.data + 2, t_count, t_scale, 0, fc_arr.idx, sizeof(Vec3d));
				ImPlot::EndPlot();
			}
		}
#endif //MEASURE_CONSTRAINT
		if (ImGui::CollapsingHeader("Options")) {
			ImGui::Checkbox("draw mesh", &show_triangle);
			ImGui::Checkbox("camera follow", &camera.should_follow);
			static float offset_min = -1;
			static float offset_max = 1;
			ImGui::DragScalarN("camera offset", ImGuiDataType_Float, &camera_target_offset, 3, 0.01, &offset_min, &offset_max, "%.2f");

			ImGui::Checkbox("use PBD", &USE_PBD);
		}
		
		ImGui::End();
		// Rendering
		ImGui::Render();
		int display_w, display_h;
		glfwGetFramebufferSize(window, &display_w, &display_h);
		glViewport(0, 0, display_w, display_h);

		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}

}

/* imgui Cleanup and shutdown */
void Simulation::shutdownImgui() {
	ImPlot::DestroyContext();
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}
/*-------------------------------------------------------------------------------*/
#endif // GRAPHICS



#ifdef UDP


/*-----------------------------------------------------------------------------------------------*/
DataSend::DataSend(
	const UDP_HEADER& header,
	const Simulation* s) : header(header), T(s->T)
{
	const auto& joint_control = s->joint_control;
	const auto& body = s->body;
	int num_joint = joint_control.size();
	joint_pos = std::vector<float>(2 * num_joint, 0);
	joint_vel = std::vector<float>(joint_control.vel, joint_control.vel + num_joint);
	joint_torque = std::vector<float>(s->joint.torque, s->joint.torque+ num_joint);

	for (auto i = 0; i < joint_control.size(); i++) {
		joint_pos[i * 2] = cosf(joint_control.pos[i]);
		joint_pos[i * 2 + 1] = sinf(joint_control.pos[i]);
	}
	body.acc.fillArray(com_acc);
	body.vel.fillArray(com_vel);
	body.pos.fillArray(com_pos);
	body.ang_vel.fillArray(ang_vel);
	// body orientation
	orientation[0] = body.rot.m00;
	orientation[1] = body.rot.m10;
	orientation[2] = body.rot.m20;
	orientation[3] = body.rot.m01;
	orientation[4] = body.rot.m11;
	orientation[5] = body.rot.m21;


#ifdef STRESS_TEST
	int NUM_SPRING_STRAIN = 0;
	if (NUM_SPRING_STRAIN > 0) {
		if (s->id_selected_edges.size() > 0 && (NUM_SPRING_STRAIN > 0)) { // only update if there selected edges exists
			int step_spring_strain = s->id_selected_edges.size() / NUM_SPRING_STRAIN;
			spring_strain = std::vector<float>(NUM_SPRING_STRAIN, 0);// initialize vector
			for (int k = 0; k < NUM_SPRING_STRAIN; k++)// set values
			{
				int i = s->id_selected_edges[k * step_spring_strain];
				Vec2i e = s->spring.edge[i];
				Vec3d s_vec = s->mass.pos[e.y] - s->mass.pos[e.x];// the vector from left to right
				double length = s_vec.norm(); // current spring length
				spring_strain[k] = (length - s->spring.rest[i]) / s->spring.rest[i];
			}
		}
	}
#endif // STRESS_TEST
	
#ifdef MEASURE_CONSTRAINT
	//TODO change it!
	float total_weight = s->total_mass * s->global_acc.norm();
	constraint_force.resize(s->body_constraint_force.size());
	for (int i = 0; i < s->body_constraint_force.size(); i++){ // normalized by total_weight
		//constraint_force[i] = s->body_constraint_force[i].norm();
		constraint_force[i] = s->body_constraint_force[i].norm()/ total_weight>0.05? 1.0:0;
	}
#endif //MEASURE_CONSTRAINT

}

#endif