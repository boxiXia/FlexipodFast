#include <iostream>
#include <iomanip>
#include <WS2tcpip.h>
#include <msgpack.hpp>
#include <vector>
#include <string>

// Include the Winsock library (lib) file
#pragma comment (lib, "ws2_32.lib")

using namespace std;
class UDPdata {
public:
	double rotationAngle[4];
	double quaternion[4];
	double acceleration[3];
	MSGPACK_DEFINE(rotationAngle, quaternion, acceleration);
};

void main()
{

	WSADATA data;

	WORD version = MAKEWORD(2, 2);

	// Start WinSock
	int wsOk = WSAStartup(version, &data);
	if (wsOk != 0)
	{
		// Not ok! Get out quickly
		cout << "Can't start Winsock! " << wsOk;
		return;
	}

	SOCKET in = socket(AF_INET, SOCK_DGRAM, 0);

	sockaddr_in serverHint;
	serverHint.sin_addr.S_un.S_addr = ADDR_ANY; // Us any IP address available on the machine
	serverHint.sin_family = AF_INET; // Address format is IPv4
	serverHint.sin_port = htons(2000); // Convert from little to big endian

	bind(in, (sockaddr*)&serverHint, sizeof(serverHint)) == SOCKET_ERROR;

	sockaddr_in client; // Use to hold the client information (port / ip address)
	int clientLength = sizeof(client); // The size of the client information

	char buf[200];
	unsigned short k;
	int output = 0;
	int UDPrecv;
	while (true)
	{

		int datalength = recvfrom(in, buf, 200, 0, (sockaddr*)&client, &clientLength);

		msgpack::object_handle oh = msgpack::unpack(buf, datalength);

		msgpack::object obj = oh.get();
		// you can convert object to UDPdata class directly
		std::vector<UDPdata> rvec;
		obj.convert(rvec);

		printf("-----------------------------------------------------------\n");
		printf("RotationAngle: ");
		for (int i = 0; i < 4; i++)
		{
			printf("%f", rvec[0].rotationAngle[i]);
			printf(" ");
		}
		printf("\n");

		printf("Quaternion: ");
		for (int i = 0; i < 4; i++)
		{
			printf("%f", rvec[0].quaternion[i]);
			printf(" ");
		}
		printf("\n");

		printf("Acceleration: ");
		for (int i = 0; i < 3; i++)
		{
			printf("%f", rvec[0].acceleration[i]);
			printf(" ");
		}
		printf("\n");
	}

	// Close socket
	closesocket(in);

	// Shutdown winsock
	WSACleanup();
}
