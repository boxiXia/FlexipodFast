// C++ UDP Transmitter
// ref: https://adaickalavan.github.io/programming/udp-socket-programming-in-cpp-and-python/
#include "Network.h"


#pragma once

void UdpReceive() {
    int port_local = 32001;

    WSASession Session;

    UDPSocket receiver_socket;
    UdpDataReceive msg_rec;
    char buffer[128];

    receiver_socket.Bind(port_local);

    while (true) {
        int n_bytes_received;
        sockaddr_in add = receiver_socket.RecvFrom(buffer, sizeof(buffer), n_bytes_received);

        msgpack::object_handle oh = msgpack::unpack(buffer, n_bytes_received);
        msgpack::object obj = oh.get();

        obj.convert(msg_rec);
        printf("%3.3f \t %3.3f %3.3f %3.3f %3.3f\r\r", msg_rec.T,
            msg_rec.jointSpeed[0],
            msg_rec.jointSpeed[1],
            msg_rec.jointSpeed[2],
            msg_rec.jointSpeed[3]);

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void UdpSend() {
    std::string ip_remote = "127.0.0.1";
    int port_remote = 32000;

    UdpDataSend msg_send;
    auto& T = msg_send.T;
    try
    {
        WSASession Session;

        UDPSocket sender_socket;
        //sender_socket.SetRemoteAddress(ip_remote, port_remote);
        while (1)
        {
            T += 0.01;
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

            std::stringstream ss;
            msgpack::pack(ss, msg_send);
            std::string const& data = ss.str();

            sender_socket.SendTo(ip_remote, port_remote, data.c_str(), data.size());
            //sender_socket.Send(data.c_str(), data.size());


            std::this_thread::sleep_for(std::chrono::milliseconds(10));

            //std::cout << "Enter data to transmit : " << std::endl;
            //std::getline(std::cin, data);
            //Socket.SendTo(ip_remote, PORT, data.c_str(), data.size());
        }
    }
    catch (std::exception& ex)
    {
        std::cout << ex.what();
    }
}
int main()
{

    std::thread thread_object_r(UdpReceive);
    std::thread thread_object_s(UdpSend);

    while (1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    thread_object_r.join();
    thread_object_s.join();
}