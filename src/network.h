// Network.h header file for C++ UDP
// copied from: https://adaickalavan.github.io/programming/udp-socket-programming-in-cpp-and-python/
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#include <system_error>
#include <string>
#include <iostream>
#include <WS2tcpip.h>

#include <msgpack.hpp>
#include "iostream"
#include <sstream>
#include <thread>
#include <time.h> // for timeout setup
#pragma once
#pragma comment (lib, "ws2_32")

class WSASession
{
public:
    WSASession()
    {
        int ret = WSAStartup(MAKEWORD(2, 2), &data);
        if (ret != 0)
            throw std::system_error(WSAGetLastError(), std::system_category(), "WSAStartup Failed");
    }
    ~WSASession()
    {
        WSACleanup();
    }

private:
    WSAData data;
};

class UDPSocket
{
public:
    //std::string ip_remote;
    //int port_remote;
    //int port_local;
    sockaddr_in remote_address;

    UDPSocket()
    {
        sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (sock == INVALID_SOCKET)
            throw std::system_error(WSAGetLastError(), std::system_category(), "Error opening socket");
    }
    ~UDPSocket()
    {
        //shutdown(sock, SD_RECEIVE);
        closesocket(sock);

    }

    /*set the remote_address give an address (e.g. "127.0.0.1"),and the prot (e.g. 5000) (by Boxi)*/
    void SetRemoteAddress(const std::string& address, unsigned short port)
    {
        remote_address.sin_family = AF_INET;
        remote_address.sin_addr.s_addr = inet_addr(address.c_str());
        remote_address.sin_port = htons(port);
    }

    void SendTo(const std::string& address, unsigned short port, const char* buffer, int len, int flags = 0)
    {
        sockaddr_in add;
        add.sin_family = AF_INET;
        add.sin_addr.s_addr = inet_addr(address.c_str());
        //add.sin_addr.s_addr = inet_pton(AF_INET, "127.0.0.1", address.c_str());
        add.sin_port = htons(port);
        int ret = sendto(sock, buffer, len, flags, reinterpret_cast<SOCKADDR*>(&add), sizeof(add));
        if (ret < 0)
            throw std::system_error(WSAGetLastError(), std::system_category(), "sendto failed");
    }

    void SendTo(sockaddr_in& address, const char* buffer, int len, int flags = 0)
    {
        int ret = sendto(sock, buffer, len, flags, reinterpret_cast<SOCKADDR*>(&address), sizeof(address));
        if (ret < 0)
            throw std::system_error(WSAGetLastError(), std::system_category(), "sendto failed");
    }

    /* send a the buffer content to the remote_address, you must call SetRemoteAddress(...) method to set the remote_address first*/
    void Send(const char* buffer, int len, int flags = 0) {
        int ret = sendto(sock, buffer, len, flags, reinterpret_cast<SOCKADDR*>(&remote_address), sizeof(remote_address));
        if (ret < 0)
            throw std::system_error(WSAGetLastError(), std::system_category(), "sendto failed");
    }


    /* ref: https://docs.microsoft.com/en-us/windows/win32/api/winsock/nf-winsock-recvfrom
    buffer: A buffer for the incoming data.
    buffer_len: The length, in bytes, of the buffer pointed to by the buf parameter.
    flags: A set of options that modify the behavior of the function call beyond the options specified for the associated socket.
    n_bytes_received: the number of bytes received
    */
    sockaddr_in RecvFrom(char* buffer, int buffer_len,int& n_bytes_received, int flags = 0)
    {
        sockaddr_in from;
        int size = sizeof(from);
        int ret = recvfrom(sock, buffer, buffer_len, flags, reinterpret_cast<SOCKADDR*>(&from), &size);
        if (ret < 0)
            throw std::system_error(WSAGetLastError(), std::system_category(), "recvfrom failed");
        // make the buffer zero terminated
        buffer[ret] = 0;
        n_bytes_received = ret;
        return from;
    }
    void Bind(unsigned short port)
    {
        sockaddr_in add;
        add.sin_family = AF_INET;
        add.sin_addr.s_addr = htonl(INADDR_ANY);
        add.sin_port = htons(port);

        int ret = bind(sock, reinterpret_cast<SOCKADDR*>(&add), sizeof(add));
        if (ret < 0)
            throw std::system_error(WSAGetLastError(), std::system_category(), "Bind failed");
    }

    void SetTimeout(int microsecond=10000) {
        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = microsecond;
        if (setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (char*)&tv, sizeof(tv)) < 0) {
            perror("Error");
        }
    }

private:
    SOCKET sock;
};



class UdpDataSend {/*the info to be sent to the high level controller*/
public:
    int header = 11; 
    double T = 0;
    double jointAngle[4] = { 0 };
    double orientation[6] = { 0 };
    double acceleration[3] = { 0 };
    double position[3] = { 0 };
    MSGPACK_DEFINE(header, T, jointAngle, orientation, acceleration, position);
};

class UdpDataReceive {/*the high level command to be received */
public:
    int header = 12;
    double T;
    double jointSpeed[4] = { 0 };
    MSGPACK_DEFINE(header, T, jointSpeed);
};

