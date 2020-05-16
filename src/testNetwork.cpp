// C++ UDP Transmitter
// ref: https://adaickalavan.github.io/programming/udp-socket-programming-in-cpp-and-python/
#include "Network.h"
#include "iostream"
#include <sstream>
#include <msgpack.hpp>
#include <thread>

#pragma once


class UdpDataSend {
public:
    int header = 11;
    double T=0;
    double jointAngle[4] = { 0 };
    double quaternion[4] = { 0 };
    double acceleration[3] = { 0 };
    MSGPACK_DEFINE(header,T,jointAngle, quaternion, acceleration);
};

class UdpDataReceive {
public:
    int header = 12;
    double T;
    double jointSpeed[4] = { 0 };
    MSGPACK_DEFINE(header,T, jointSpeed);
};


void UdpReceive() {
    int port_local = 32001;

    WSASession Session;

    UDPSocket receiver_socket;
    UdpDataReceive msg_rec;
    char buffer[100];

    receiver_socket.Bind(port_local);
    
    while (true) {
        int n_bytes_received;
        sockaddr_in add = receiver_socket.RecvFrom(buffer, sizeof(buffer), n_bytes_received);

        msgpack::object_handle oh = msgpack::unpack(buffer, n_bytes_received);
        msgpack::object obj = oh.get();

        obj.convert(msg_rec);
        printf("%f", msg_rec.T);

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
int main()
{
    std::string ip_remote = "127.0.0.1";
    int port_remote = 32000;
        
    UdpDataSend msg_send;
    auto& T = msg_send.T;

    
    try
    {
        WSASession Session;

        UDPSocket sender_socket;

        std::thread threadObj(UdpReceive);
        
        while (1)
        {
            T += 0.01;
            for (int i = 0; i < 4; i++)
            {
                msg_send.jointAngle[i] = sin(T+i);
            }


            std::stringstream ss;
            msgpack::pack(ss, msg_send);
            std::string const& data = ss.str();

            sender_socket.SendTo(ip_remote, port_remote, data.c_str(), data.size());


            std::this_thread::sleep_for(std::chrono::milliseconds(10));

            //std::cout << "Enter data to transmit : " << std::endl;
            //std::getline(std::cin, data);
            //Socket.SendTo(ip_remote, PORT, data.c_str(), data.size());
        }

        threadObj.join();
    }
    catch (std::exception& ex)
    {
        std::cout << ex.what();
    }

    
}