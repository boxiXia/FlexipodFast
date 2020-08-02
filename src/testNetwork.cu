#include "Network.h"
#include <iostream>
#include <vector>
#include <string>
#include<thread>

#include <conio.h>//keyboard interrupt

#include <cuda_runtime.h>

typedef WsaUdpServer UdpServer;
//typedef AsioUdpServer UdpServer;


class TestUdp {
public:
	UdpServer s;
	std::thread control_thread;

	void controlLoop() {

		s.run();

		UdpDataSend& msg_send = s.msg_send;
		auto& T = msg_send.T;

		UdpDataReceive& msg_rec = s.msg_rec;

		while (1) {

			// Data for testing UDP
			double T = std::chrono::system_clock::now().time_since_epoch().count();

			for (int i = 0; i < 4; i++)
			{
				msg_send.jointAngle[i] = sin(T + i);
				msg_send.orientation[i] = cos(T + i);
			}
			for (size_t i = 0; i < 3; i++)
			{
				msg_send.acceleration[i] = tan(T + i);
				msg_send.position[i] = sin(T + i + 1);
			}
			s.flag_should_send = true;
			std::this_thread::sleep_for(std::chrono::milliseconds(10));


			if (s.flag_new_received) {//new massg
				printf("%7.3f \t %3.3f %3.3f %3.3f %3.3f\n", msg_rec.T,
					msg_rec.jointSpeed[0],
					msg_rec.jointSpeed[1],
					msg_rec.jointSpeed[2],
					msg_rec.jointSpeed[3]);
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
		printf("close window");
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
	}
	//control_thread.join();


	return 0;
}