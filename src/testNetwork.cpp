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
    double T=0;
    double jointAngle[4] = { 0 };
    double quaternion[4] = { 0 };
    double acceleration[3] = { 0 };
    MSGPACK_DEFINE(T,jointAngle, quaternion, acceleration);
};

class UdpDataReceive {
public:
    double T;
    double jointSpeed[4] = { 0 };
    MSGPACK_DEFINE(T, jointSpeed);
};

int main()
{
    std::string IP = "127.0.0.1";
    int PORT = 32000;

    UdpDataSend msg_send;
    auto& T = msg_send.T;
    

    
    try
    {
        WSASession Session;
        UDPSocket Socket;
        char buffer[100];
        
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

            Socket.SendTo(IP, PORT, data.c_str(), data.size());
            std::this_thread::sleep_for(std::chrono::milliseconds(10));

            //std::cout << "Enter data to transmit : " << std::endl;
            //std::getline(std::cin, data);
            //Socket.SendTo(IP, PORT, data.c_str(), data.size());
        }
    }
    catch (std::exception& ex)
    {
        std::cout << ex.what();
    }
}