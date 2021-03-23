#include "Network.h"
#include <iostream>
#include <vector>
#include <string>
#include<thread>

#include <conio.h>//keyboard interrupt

#include <cuda_runtime.h>

#include "vec.h"



enum UDP_HEADER :int {
	TERMINATE = -1,// close the program
	PAUSE = 17,
	RESUME = 16,
	RESET = 15,
	ROBOT_STATE_REPORT = 14,
	MOTOR_SPEED_COMMEND = 13,
	STEP_MOTOR_SPEED_COMMEND = 12,
	MOTOR_POS_COMMEND = 11,
	STEP_MOTOR_POS_COMMEND = 10,
};
MSGPACK_ADD_ENUM(UDP_HEADER); // msgpack macro,refer to https://github.com/msgpack/msgpack-c/blob/cpp_master/example/cpp03/enum.cpp

class DataSend {/*the info to be sent to the high level controller*/
public:
	UDP_HEADER header = UDP_HEADER::ROBOT_STATE_REPORT;
	double T = 0; // time at the sending of this udp packet
	std::vector<double> joint_pos; // joint position (angles)
	std::vector<double> joint_vel; // joint angular velocity
	std::vector<double> joint_act; // acutation of the joint

	double orientation[6] = { 0 }; // orientation of the body
	Vec3d ang_vel = Vec3d(); // angular velocity of the body

	Vec3d com_acc = Vec3d();// COM (body) acceleration
	Vec3d com_vel = Vec3d(); // COM (body) velocity
	Vec3d com_pos = Vec3d(); // COM (body) position
	//double joint_vel_desired[4] = { 0 };// desired joint velocity at last command
	//double T_prev = 0; // time at last command
	MSGPACK_DEFINE_ARRAY(header, T, joint_pos, joint_vel, joint_act, orientation, ang_vel, com_acc, com_vel, com_pos);

	DataSend(){}// defualt constructor
	DataSend(int num_joint) {
		init(num_joint);
	}

	void init(int num_joint) {
		joint_pos = std::vector<double>(num_joint, 0);
		joint_vel = std::vector<double>(num_joint, 0);
		joint_act = std::vector<double>(num_joint, 0);
	}
};

class DataReceive {/*the high level command to be received */
public:
	UDP_HEADER header = UDP_HEADER::MOTOR_SPEED_COMMEND;
	double T;
	std::vector<double> joint_vel_desired;// desired joint velocity
	MSGPACK_DEFINE_ARRAY(header, T, joint_vel_desired);

	DataReceive(int num_joint) {
		init(num_joint);
	}
	DataReceive(){}// defualt constructor
	void init(int num_joint) {
		joint_vel_desired = std::vector<double>(num_joint, 0);
	}
};

typedef WsaUdpServer< DataReceive, DataSend> UdpServer;




class TestUdp {
public:
	UdpServer s;
	std::thread control_thread;

	TestUdp(int num_joint=4):
	s(32001, 32000, "127.0.0.1")// port_local,port_remote,ip_remote,num_joint
	{
		s.msg_send.init(num_joint);
		s.msg_rec.init(num_joint);
	}

	void controlLoop() {

		s.run();

		DataSend& msg_send = s.msg_send;
		auto& T = msg_send.T;

		DataReceive& msg_rec = s.msg_rec;

		while (1) {

			// Data for testing UDP
			double T = std::chrono::system_clock::now().time_since_epoch().count();

			for (int i = 0; i < 4; i++)
			{
				msg_send.joint_pos[i] = sin(T + i);
				msg_send.orientation[i] = cos(T + i);
			}
			for (size_t i = 0; i < 3; i++)
			{
				msg_send.com_acc[i] = tan(T + i);
				msg_send.com_pos[i] = sin(T + i + 1);
			}
			s.flag_should_send = true;
			std::this_thread::sleep_for(std::chrono::milliseconds(10));


			if (s.flag_new_received) {//new massg
				printf("%7.3f \t %3.3f %3.3f %3.3f %3.3f\n", msg_rec.T,
					msg_rec.joint_vel_desired[0],
					msg_rec.joint_vel_desired[1],
					msg_rec.joint_vel_desired[2],
					msg_rec.joint_vel_desired[3]);
				s.flag_new_received = false;
			}

			if (_kbhit())
			{
				char key = _getch();
				if (key == 27) {
					s.flag_should_close = true;
					break;
				}
			}


		}
		printf("close window\n");
	}


	void run() {
		control_thread = std::thread(&TestUdp::controlLoop, this);
	}
};


int main()
{
	TestUdp test_udp;
	test_udp.run();


	while (1) {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		if (test_udp.control_thread.joinable()) {
			break;
		}
	}
	test_udp.control_thread.join();
	test_udp.s.close();


	return 0;
}