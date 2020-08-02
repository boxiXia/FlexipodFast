// Network.h header file for C++ UDP
// copied from: https://adaickalavan.github.io/programming/udp-socket-programming-in-cpp-and-python/
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#include <system_error>
#include <string>
#include <iostream>
#include <WS2tcpip.h>

#include <asio.hpp>
#include <msgpack.hpp>
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

class UdpServer {
public:
    UdpDataReceive msg_rec; // struct to be received
    UdpDataSend msg_send; // struct to be sent

    int port_local; // local port
    int port_remote; // remote port
    std::string ip_remote; // remote ip

    bool send_flag = false;// flag indicating whether to send udp packet
    bool should_close = false; // flag indicating whether to stop sending/receiving
    asio::io_context io_context; // asio io context for the socket
    std::thread udp_send_thread; // thread for sending udp
    std::thread udp_receive_thread; // thread for receiving udp

    /*constructor*/
    UdpServer(
        int port_local = 32001,
        int port_remote = 32000,
        std::string ip_remote = "127.0.0.1")
        :socket(io_context, asio::ip::udp::endpoint(asio::ip::udp::v4(), port_local)) {
        this->port_remote = port_remote;
        this->port_local = port_local;
        this->ip_remote = ip_remote;
        remote_endpoint = asio::ip::udp::endpoint(asio::ip::address::from_string(ip_remote), port_remote);
        socket.connect(remote_endpoint);// connect to remote_endpoint
    }
    /* loop for receiving the udp packet */
    void do_receive()
    {
        while (!should_close) {
            socket.async_receive(// receive from remote_endpoint
                asio::buffer(recv_buffer_, max_length),
                [this](std::error_code ec, std::size_t bytes_recvd)
                {
                    if (!ec && bytes_recvd > 0)
                    {
                        //for (int i = 0; i < bytes_recvd; i++) {//prints the data in hex format
                        //	printf("%02x", reinterpret_cast<unsigned char*>(recv_buffer_)[i]);
                        //}
                        //printf("\n");

                        // Unpack data
                        msgpack::object_handle oh = msgpack::unpack(recv_buffer_, bytes_recvd);
                        msgpack::object obj = oh.get();
                        obj.convert(msg_rec);

                        // Check data
                        obj.convert(msg_rec);
                        printf("%7.3f \t %3.3f %3.3f %3.3f %3.3f\n", msg_rec.T,
                            msg_rec.jointSpeed[0],
                            msg_rec.jointSpeed[1],
                            msg_rec.jointSpeed[2],
                            msg_rec.jointSpeed[3]);
                        //TODO notify the simulation thread
                    }
                });
            io_context.run_for(std::chrono::duration<int, std::milli>(100));
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }

    }

    /* loop for sending the udp packet */
    void do_send()
    {
        try {
            while (!should_close) {
                while (!send_flag) { std::this_thread::sleep_for(std::chrono::microseconds(100)); }
                // Pack data into msgpack
                std::stringstream ss;
                msgpack::pack(ss, msg_send);
                socket.send(asio::buffer(ss.str()));//send to remote_endpoint
                //std::string const& data = ss.str();
                //socket.send(asio::buffer(data.c_str(), data.size()));//send to remote_endpoint
                send_flag = false;//reset send_flag
            }
        }
        catch (asio::error_code e) {

        }

    }
    /*run this to start receiving and sending udp*/
    void run() {
        //udp_receive_thread = std::thread{ &UdpServer::do_receive, this };
        udp_send_thread = std::thread(&UdpServer::do_send, this);
    }

    ~UdpServer() {
        udp_send_thread.join();
        udp_receive_thread.join();
        io_context.stop();
        socket.close();
    }
private:

    asio::ip::udp::socket socket;//local socket for sending and reciving udp packets
    asio::ip::udp::endpoint remote_endpoint;
    enum { max_length = 1024 };
    char recv_buffer_[max_length];// data received from the remote endponit
    char send_buffer_[max_length];// data to be sent to the remote endpoint
};