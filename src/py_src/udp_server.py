"""
udp_server.py: helper class for sending and receving udp messages, support unicast/multicast ipv4/ipv6

"""
__author__ = "Boxi Xia"
__license__ = "Apache"

import numpy as np
import socket
import struct
from ipaddress import ip_address # https://python.readthedocs.io/en/latest/library/ipaddress.html
import time

class UDPServer:
    def __init__(
        s,  # self
        local_address=("224.3.29.71", 33300), # （ipv4/6 local address, local port)
        remote_address=("224.3.29.71", 33301), #（ipv4/6 remote address, remote port)
        ttl:int = 1, # time-to-live, increase to reach beyond local network
        buffer_len:int = 65536, # buffer length in bytes
    ):
        s.BUFFER_LEN = buffer_len  # in bytes
        s._local_address = local_address
        s._remote_address = remote_address
        s.ttl = ttl
        s._start()

    def _start(s):
        # udp socket for sending
        s._makeSendScoket()
        # udp socket for receving
        s._makeRecvSocket()

    def _makeSendScoket(s):
        """prepare udp socket for sending"""
        print(s._remote_address)
        (s.send_family, type, proto, canonname, s.remote_address) = socket.getaddrinfo(*s._remote_address)[0]
        s.send_sock = socket.socket(family=s.send_family, type=socket.SOCK_DGRAM)
        ttl_bin = struct.pack('@i', s.ttl) # time-to-live, increase to reach beyond local networkt
        if s.send_family== socket.AF_INET: # IPv4
            s.send_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl_bin)
        else: # IPv6
            s.send_sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_MULTICAST_HOPS, ttl_bin)

    def _makeRecvSocket(s):
        """prepare udp socket for receiving"""
        # udp socket for receving
        (s.recv_family, type, proto, canonname, s.local_address) = socket.getaddrinfo(*s._local_address)[0]
        s.recv_sock = socket.socket(family=s.recv_family, type=socket.SOCK_DGRAM)
        if ip_address(s.local_address[0]).is_multicast:
            group_bin = socket.inet_pton(s.recv_family,s.local_address[0]) # multicast address in bytes
            mreq = group_bin + struct.pack('=I', socket.INADDR_ANY)
            if s.recv_family== socket.AF_INET: # IPv4
                s.recv_sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            else: # IPv6
                s.recv_sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_JOIN_GROUP, mreq)
        s.recv_sock.bind(("",s.local_address[1]))  # Bind socket to local port


    def clearRecvBuffer(s):
        """clear old messages that exits in receive socket buffer"""
        timeout = s.recv_sock.gettimeout() # original timeout
        s.recv_sock.settimeout(0) # timeout immediately
        try:
            while True:
                _ = s.recv_sock.recv(s.BUFFER_LEN)
        except Exception:
            pass
        s.recv_sock.settimeout(timeout)
        

    def receive(s,
                timeout: float = 1, # timeout [seconds]
                clear_buf: bool = True,
                newest: bool = True):
        """ return the recived data at local port
            Input:
                timeout: [second] longest duration to wait before failing
                clear_buff: bool, if True clear messages prior to receive()
                newest: bool, if True only get the newest data after receive() is called
        """
        # clear receive socket buffer, any old messages prior to
        # receive will be cleared


        if clear_buf or newest:
            s.recv_sock.settimeout(0) # timeout immediately
            try:
                while True:
                    recv_data = s.recv_sock.recv(s.BUFFER_LEN)
            except Exception as e:
                if newest:
                    s.recv_sock.settimeout(timeout)
                    return s.recv_sock.recv(s.BUFFER_LEN)
                else:
                    try:
                        return recv_data
                    except NameError: # data is not received
                        s.recv_sock.settimeout(timeout)
                        return s.recv_sock.recv(s.BUFFER_LEN)
        else:
            s.recv_sock.settimeout(timeout)
            return s.recv_sock.recv(s.BUFFER_LEN)


    def send(s, data):
        """send the data to remote address, return num_bytes_send"""
        return s.send_sock.sendto(data, s.remote_address)

    def close(s):
        try:
            s.recv_sock.shutdown(socket.SHUT_RDWR)
            s.recv_sock.close()
        except Exception as e:
            print(e)
        print(f"shutdown UDP server:{s._local_address},{s._remote_address}")

    def __del__(s):
        s.close()
        
        
if __name__ == '__main__':
    # Usage:
    # run as server:
        # python udp_server -s
    # run as client:
        # python udp_server
    # optional command:
        # [-s]  : if True run as server, else run as client
        # [-v6] : if True use ipv6 address, else use ipv4 address
        # [-m]  : if True use multicast, else unicast
        
    # ref: https://svn.python.org/projects/python/trunk/Demo/sockets/mcast.py
    import argparse
    parser = argparse.ArgumentParser(description='udp server demo')
    parser.add_argument('-s',action = 'store_true', help="as server")
    parser.add_argument('-v6',action = 'store_true', help="ipv6")
    parser.add_argument('-m',action = 'store_true', help="multicast")
    

    args = parser.parse_args()
    
    if args.v6: # ipv6
        if args.m: # multicase
            host = 'ff31::8000:1234' # multicast IPv6
        else:
            host = '::1' # unicast IPv6
    else: # ipv4
        if args.m:
            host = '224.3.29.71' # multicast IPv4
        else:
            host = '127.0.0.1' # unicast IPv4
    # 
    # host = '127.0.0.1' # unicast IPv4
    # host = '::1' # unicast IPv6
    ports = (30001,30002)
    
    if args.s: # as server
        print("as server")
        s = UDPServer( local_address=(host, ports[0]),remote_address=(host, ports[1]))
        for k in range(int(1e8)):
            data = s.receive(clear_buf=False,newest=False)
            print(data)
            s.send(f's->r {k}'.encode('utf-8'))
    else: # as clinet
        print("as client")
        s = UDPServer( local_address=(host, ports[1]),remote_address=(host, ports[0]))
        for k in range(int(1e8)):
            s.send(f's->r {k}'.encode('utf-8'))
            data = s.receive(clear_buf=False,newest=False)
            print(data)

    