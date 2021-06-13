#include "sim.h"

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

// Implementing a simple custom widget using the public API.
// You may also use the <imgui_internal.h> API to get raw access to more data/helpers, however the internal API isn't guaranteed to be forward compatible.
// FIXME: Need at least proper label centering + clipping (internal functions RenderTextClipped provides both but api is flaky/temporary)
static bool MyKnob(const char* label, float* p_value, float v_min, float v_max)
{
	ImGuiIO& io = ImGui::GetIO();
	ImGuiStyle& style = ImGui::GetStyle();

	float radius_outer = 40.0f;
	ImVec2 pos = ImGui::GetCursorScreenPos();
	ImVec2 center = ImVec2(pos.x + radius_outer, pos.y + radius_outer);
	float line_height = ImGui::GetTextLineHeight();
	ImDrawList* draw_list = ImGui::GetWindowDrawList();

	float ANGLE_MIN = 3.141592f * -1.0f;
	float ANGLE_MAX = 3.141592f * 1.0f;

	ImGui::InvisibleButton(label, ImVec2(radius_outer * 2, radius_outer * 2 + line_height + style.ItemInnerSpacing.y));
	bool value_changed = false;
	bool is_active = ImGui::IsItemActive();
	bool is_hovered = ImGui::IsItemActive();
	if (is_active && ((io.MouseDelta.x != 0.0f) || (io.MouseDelta.y != 0.0f)))
	{

		float step = (v_max - v_min) / 200.0f;
		//*p_value += io.MouseDelta.x * step;
		//*p_value += atan2f(io.MouseDelta.y, io.MouseDelta.x)* step;
		*p_value += (io.MouseDelta.y + io.MouseDelta.x) * step;

		if (*p_value < v_min) *p_value = v_min;
		if (*p_value > v_max) *p_value = v_max;
		value_changed = true;
	}

	float t = (*p_value - v_min) / (v_max - v_min);
	float angle = ANGLE_MIN + (ANGLE_MAX - ANGLE_MIN) * t;
	float angle_cos = cosf(angle), angle_sin = sinf(angle);
	float radius_inner = radius_outer * 0.20f;
	draw_list->AddCircleFilled(center, radius_outer, ImGui::GetColorU32(ImGuiCol_FrameBg), 16);
	draw_list->AddLine(ImVec2(center.x + angle_cos * radius_inner, center.y + angle_sin * radius_inner), ImVec2(center.x + angle_cos * (radius_outer - 2), center.y + angle_sin * (radius_outer - 2)), ImGui::GetColorU32(ImGuiCol_SliderGrabActive), 2.0f);
	draw_list->AddCircleFilled(center, radius_inner, ImGui::GetColorU32(is_active ? ImGuiCol_FrameBgActive : is_hovered ? ImGuiCol_FrameBgHovered : ImGuiCol_FrameBg), 16);
	draw_list->AddText(ImVec2(pos.x, pos.y + radius_outer * 2 + style.ItemInnerSpacing.y), ImGui::GetColorU32(ImGuiCol_Text), label);

	if (is_active || is_hovered)
	{
		ImGui::SetNextWindowPos(ImVec2(pos.x - style.WindowPadding.x, pos.y - line_height - style.ItemInnerSpacing.y - style.WindowPadding.y));
		ImGui::BeginTooltip();
		ImGui::Text("%.3f", *p_value);
		ImGui::EndTooltip();
	}

	return value_changed;
}

/*Setup Dear ImGui*/
void Simulation::startupImgui() {
	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	//ImGui::StyleColorsClassic();

	// fonts
	//io.Fonts->AddFontDefault();
	//io.Fonts->TexDesiredWidth = 20;
	//ImGui::SetWindowFontScale(2);
	//ImGui::GetFont()->FontSize = 20;

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
	float xscale=2, yscale=2;
	//glfwGetMonitorContentScale(monitor, &xscale, &yscale);
	//std::cout << xscale << "," << yscale;
	std::string font_path = (getProgramDir() + "\\Cousine-Regular.ttf");

	io.Fonts->AddFontFromFileTTF(font_path.c_str(), 16.0f * xscale);
	ImGui::GetStyle().ScaleAllSizes(xscale);

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
		//bool show_demo_window = true;
		//ImGui::ShowDemoWindow(&show_demo_window);
		//ImGui::ShowMetricsWindow();
		//ImGui::ShowStyleEditor();

		// measure simulation speed
		auto t = std::chrono::steady_clock::now();
		float duration = (float)std::chrono::duration_cast<std::chrono::milliseconds>(t - t_prev).count() / 1000.;//[seconds]
		if (duration > 0.3) {
			float sim_duration = T - t_sim_prev;
			sim_speed = sim_duration / duration;
			rec_fps = (float(udp_server.counter_rec - counter_rec)) / sim_duration; // frame per simulation seconds
			counter_rec = udp_server.counter_rec;
			t_sim_prev = T;
			t_prev = t;
		}

		ImGui::Begin("Debug console", &show_imgui);

		// simulation time | simulation speed | rendering FPS
		ImGui::Text("%.2f s | % 5.2f X | %.1f FPS", T, sim_speed,ImGui::GetIO().Framerate);
		ImGui::Text("UDP rec %.2f FPSS", rec_fps); 

		ImGui::Text("F_constraint: %+6.1f %+6.1f %+6.1f N", force_constraint.x, force_constraint.y, force_constraint.z);


		if (ImGui::Button("Reset")) { RESET = true; SHOULD_RUN = true; }// reset state
		ImGui::SameLine();

		if (RUNNING) { if (ImGui::Button("Pause ")) { pause(0); } } // pause
		else if (ImGui::Button("Resume")) { resume(); }// resume

		//static float v_knob = 0;
		//MyKnob("knob", &v_knob, -3.142, 3.14

		// physics
		if (joint_control.size() > 0 && ImGui::CollapsingHeader("physics")) {

			static double dt_min = 1e-7;
			static double dt_max = 1e-3;
			ImGui::DragScalar("dt", ImGuiDataType_Double, &dt, 1e-7, &dt_min, &dt_max, "%5.3e");
			ImGui::DragScalarN("gravity", ImGuiDataType_Double, &global_acc, 3, 0.1,&gravity_min, &gravity_max, "%.2f");
		}

		//ImGui::PlotLines
		// 

		// joint control
		if (joint_control.size() > 0 && ImGui::CollapsingHeader("joint control")) {

			float width = ImGui::GetContentRegionAvail().x;
			float cursor_pos_x = ImGui::GetCursorPosX();
			ImGui::Text("id");ImGui::SameLine();

			ImGui::SetCursorPosX(width * 0.2f);
			ImGui::Text("pos_desired"); ImGui::SameLine();
			ImGui::SetCursorPosX(width * 0.6f);
			ImGui::Text("vel_desired"); 
			
			ImGui::PushItemWidth(width * 0.5);
			char label[20];
			for (int i = 0; i < joint_control.size(); i++)
			{
				ImGui::Text("%2d", i); 
				ImGui::SameLine();
				
				sprintf(label, "joint_pos_des_%d", i); 
				ImGui::PushID(label);
				//ImGui::Text("%+4.3f\t", joint_control.pos_desired[i]);
				ImGui::DragScalar("", ImGuiDataType_Double, &(joint_control.pos_desired[i]), 0.005f, NULL, NULL, "%6.3f");
				ImGui::PopID();

				ImGui::SameLine();
				sprintf(label, "joint_vel_des_%d", i);
				ImGui::PushID(label);
				//ImGui::Text("%+4.3f", joint_control.vel_desired[i]);
				ImGui::DragScalar("", ImGuiDataType_Double, &(joint_control.vel_desired[i]), 0.005f, NULL, NULL, "%6.3f");
				ImGui::PopID();
			}
			ImGui::PopItemWidth();
			ImGui::Separator();

			// ref: https://github.com/ocornut/imgui/blob/838c16533d3a76b83f0ca73045010d463b73addf/imgui_demo.cpp#L687
			const char* elem_name = (joint_control.mode == JointControlMode::vel)?  "vel":"pos";
			ImGui::SliderInt("control mode", &((int&)joint_control.mode), 0, 1, elem_name);
			ImGui::Text("com pos %+6.2f %+6.2f %+6.2f", body.pos.x, body.pos.y, body.pos.z);
		}

		if (ImGui::CollapsingHeader("options")) {
			ImGui::Checkbox("draw mesh ", &show_triangle);
			ImGui::Checkbox("camera follow ", &camera.should_follow);
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
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}
/*-------------------------------------------------------------------------------*/
#endif // GRAPHICS