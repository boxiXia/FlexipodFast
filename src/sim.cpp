#include "sim.h"

#ifdef GRAPHICS
#include "imgui.h" // imgui
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h" // implot
#endif

#ifdef UDP
#include "asio.hpp"
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


//// utility structure for realtime plot
//struct RollingBuffer {
//	float Span;
//	ImVector<ImVec2> Data;
//	RollingBuffer() {
//		Span = 10.0f;
//		Data.reserve(2000);
//	}
//	void AddPoint(float x, float y) {
//		float xmod = fmodf(x, Span);
//		if (!Data.empty() && xmod < Data.back().x)
//			Data.shrink(0);
//		Data.push_back(ImVec2(xmod, y));
//	}
//};


/*--------------------------------- ImGui ----------------------------------------*/

// Implementing a simple custom widget using the public API.
// You may also use the <imgui_internal.h> API to get raw access to more data/helpers, however the internal API isn't guaranteed to be forward compatible.
// FIXME: Need at least proper label centering + clipping (internal functions RenderTextClipped provides both but api is flaky/temporary)
static bool MyKnob(const char* label, float* p_value, float v_min, float v_max)
{
	ImGuiIO& io = ImGui::GetIO();

	//io.SetClipboardTextFn
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
	//float xscale=2, yscale=2;
	float xscale, yscale;
	glfwGetMonitorContentScale(monitor, &xscale, &yscale);
	//std::cout << xscale << "," << yscale;
	std::string font_path = (getProgramDir() + "\\Cousine-Regular.ttf");

	io.Fonts->AddFontFromFileTTF(font_path.c_str(), 16.0f * xscale);
	// set style
	auto& style = ImGui::GetStyle();
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

		// simulation time | simulation speed | rendering FPS
		ImGui::Text("%.2f s | % 5.2f X | %.1f FPS", T, sim_speed, ImGui::GetIO().Framerate);
#ifdef UDP	
		ImGui::Text("UDP rec %.2f FPSS", rec_fps);
#endif // UDP

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
			ImGui::DragScalarN("gravity", ImGuiDataType_Double, &global_acc, 3, 0.1, &gravity_min, &gravity_max, "%.2f");
		}

		//ImGui::PlotLines
		// 

		// joint control
		if (joint_control.size() > 0 && ImGui::CollapsingHeader("Joint control")) {

			float width = ImGui::GetContentRegionAvail().x;
			float cursor_pos_x = ImGui::GetCursorPosX();
			ImGui::Text("id"); ImGui::SameLine();
			HelpMarker("   x   ", "position [rad]");

			ImGui::SameLine();
			HelpMarker("   x_d   ", "position desired [rad]");

			ImGui::SameLine(); ImGui::SetCursorPosX(width * 0.7f);
			HelpMarker("   v_d   ", "velocity desired [rad/s]");

			ImGui::PushItemWidth(width * 0.3f);

			char label[20];
			for (int i = 0; i < joint_control.size(); i++)
			{
				ImGui::Text("%2d  %+6.3f", i, joint_control.pos[i]);
				ImGui::SameLine();

				sprintf(label, "joint_pos_des_%d##00349234", i);
				ImGui::PushID(label);
				//ImGui::Text("%+4.3f\t", joint_control.pos_desired[i]);
				ImGui::DragScalar("", ImGuiDataType_Double, &(joint_control.pos_desired[i]), 0.001f, NULL, NULL, "%6.3f");
				ImGui::PopID();

				ImGui::SameLine();
				sprintf(label, "joint_vel_des_%d##00349234", i);
				ImGui::PushID(label);
				//ImGui::Text("%+4.3f", joint_control.vel_desired[i]);
				ImGui::DragScalar("", ImGuiDataType_Double, &(joint_control.vel_desired[i]), 0.005f, NULL, NULL, "%6.3f");
				ImGui::PopID();
			}
			ImGui::PopItemWidth();
			ImGui::Separator();

			// ref: https://github.com/ocornut/imgui/blob/838c16533d3a76b83f0ca73045010d463b73addf/imgui_demo.cpp#L687
			const char* elem_name = (joint_control.mode == JointControlMode::vel) ? "vel" : "pos";
			ImGui::SliderInt("control mode", &((int&)joint_control.mode), 0, 1, elem_name);
		
		}
		if (ImGui::CollapsingHeader("Info")) {
			auto info = body.print();
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
		/*	static float history = fc_arr.num * t_scale;
			ImGui::SliderFloat("History", &history, t_scale, fc_arr.num * t_scale, "%.1f s");
			int t_count = history / t_scale;
			ImPlot::SetNextPlotLimitsX(t_scale, history, ImGuiCond_Always);*/
			ImPlot::SetNextPlotLimitsX(0, std::max(fc_arr.num,1) * t_scale, ImGuiCond_Always);
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


void asioUdpServer::_doReceive()
{
	// receiver
	asio::io_context io_context;
	asio::ip::udp::socket socket_recv(io_context); // receiver socket

	asio::ip::address local_address = asio::ip::make_address(ip_local);
	asio::ip::udp::endpoint listen_endpoint(local_address.is_v4() ?
		asio::ip::udp::v4() : asio::ip::udp::v6(), port_local);
	socket_recv.open(listen_endpoint.protocol());
	socket_recv.set_option(asio::ip::udp::socket::reuse_address(true));
	if (local_address.is_multicast()) { // Join the multicast group.
		socket_recv.set_option(asio::ip::multicast::join_group(local_address));
	}
	// set timeout [ms] without using async function,  
	// ref: https://newbedev.com/how-to-set-a-timeout-on-blocking-sockets-in-boost-asio
	// asio async offical example: 
	// https://github.com/chriskohlhoff/asio/blob/master/asio/src/examples/cpp11/timeouts/blocking_udp_client.cpp
	socket_recv.set_option(asio::detail::socket_option::integer<SOL_SOCKET, SO_RCVTIMEO>{ 100 });
	socket_recv.bind(listen_endpoint);

	// buffer
	const int buffer_size = 1024;
	std::array<char, buffer_size> data_recv = { 0 };
	auto buffer_recv = asio::mutable_buffer(data_recv.data(), data_recv.size());

	// reset counter and flags
	flag_new_received = false;
	counter_rec = 0;

	// loop
	while (UDP_SHOULD_RUN) {
		try {
			size_t nbytes = socket_recv.receive(buffer_recv);
			if (nbytes > 0) {
				msg_recv_queue.emplace_back(std::string(data_recv.data(), nbytes)); // push to msg queue
				flag_new_received = true;
				counter_rec++;
				//std::cout.write(data_recv.data(), nbytes);
				//std::cout << std::endl;
			}
		}
		catch (asio::system_error& e) { // timed out
			//printf("Caught exception %s,line %d: %s \n", __FILE__, __LINE__, e.what());
			continue;
		}
		catch (const std::exception& e) {
			printf("Caught exception %s,line %d: %s \n", __FILE__, __LINE__, e.what());
		}
		//try {
		//	// do something with the obj
		//	while (msg_recv_queue.size() > 0) {
		//		auto& data_recv = msg_recv_queue.front(); // oldest message
		//		// Unpack data
		//		msgpack::object_handle oh = msgpack::unpack(data_recv.data(), data_recv.size());
		//		msgpack::object obj = oh.get();
		//		std::cout << obj << std::endl;
		//		//std::cout << obj.as<int>() << std::endl;
		//		msg_recv_queue.pop_front();
		//		flag_new_received = false;
		//	}
		//}
		//catch (const std::bad_cast& e) { // msgpack obj.as<T> error
		//	printf("Error converting %s,line %d: %s \n", __FILE__, __LINE__, e.what());
		//}

		//sim->updateUdpReceive();
	}
}

void asioUdpServer::_doSend()
{
	asio::io_context io_context;
	asio::ip::udp::socket socket_send(io_context); // receiver socket
	// sender
	asio::ip::udp::endpoint remote_endpoint = asio::ip::udp::endpoint(asio::ip::make_address(ip_remote), port_remote);
	socket_send.open(remote_endpoint.protocol());
	socket_send.connect(remote_endpoint); // "connect" to remote endpoint

	while (UDP_SHOULD_RUN) {
		// do gether msg to send to msg_send_queue here...
		//msg_send_queue.emplace_back("message");
		//std::this_thread::sleep_for(std::chrono::milliseconds(100));
		// 
		//sim->updateUdpSend();

		while (msg_send_queue.size() > 0) {
			auto& message_ = msg_send_queue.front(); // oldest message
			std::size_t length = socket_send.send(asio::const_buffer(message_.data(), message_.size()));
			//std::size_t length = socket_send.send_to(asio::buffer(message_), remote_endpoint);
			msg_send_queue.pop_front();
		}
	}
}




/*-----------------------------------------------------------------------------------------------*/
DataSend::DataSend(
	const UDP_HEADER& header,
	const Simulation* s) : header(header), T(s->T)
{
	const auto& joint_control = s->joint_control;
	const auto& body = s->body;
	int num_joint = joint_control.size();
	joint_pos = std::vector<float>(2 * num_joint, 0);
	joint_vel = std::vector<float>(joint_control.vel, joint_control.vel + joint_control.size());
	joint_act = std::vector<float>(num_joint, 0);

	//joint_pos = std::vector<float>(joint_control.pos, joint_control.pos + joint_control.size());

	for (auto i = 0; i < joint_control.size(); i++) {
		joint_pos[i * 2] = cosf(joint_control.pos[i]);
		joint_pos[i * 2 + 1] = sinf(joint_control.pos[i]);
		joint_act[i] = joint_control.cmd[i] / joint_control.max_vel;//normalize		
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
	constexpr int NUM_SPRING_STRAIN = 0;
#if NUM_SPRING_STRAIN>0:
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
#endif
#endif // STRESS_TEST
	
#ifdef MEASURE_CONSTRAINT
	//TODO change it!
	float total_weight = s->total_mass * s->global_acc.norm();
	constraint_force.resize(s->body_constraint_force.size());
	for (int i = 0; i < s->body_constraint_force.size(); i++){ // normalized by total_weight
		constraint_force[i] = s->body_constraint_force[i].norm()/ total_weight;
	}
#endif //MEASURE_CONSTRAINT

}

#endif