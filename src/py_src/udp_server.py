"""
udp_server.py: helper class for sending and receving udp messages, support unicast/multicast ipv4/ipv6

"""
__author__ = "Boxi Xia"
__license__ = "Apache"

import numpy as np
import socket
import struct
from ipaddress import ip_address # https://python.readthedocs.io/en/latest/library/ipaddress.html

class UDPServer:
    def __init__(
        s,  # self
        local_address=("224.3.29.71", 33300), # （ipv4/6 local address, local port)
        remote_address=("224.3.29.71", 33301), #（ipv4/6 remote address, remote port)
        ttl:int = 1, # time-to-live, increase to reach beyond local network
        buffer_len:int = 65536 # buffer length in bytes
    ):
        s.BUFFER_LEN = buffer_len  # in bytes

        # udp socket for sending
        (family, type, proto, canonname, s.remote_address) = socket.getaddrinfo(*remote_address)[0]
        s.send_sock = socket.socket(family=family, type=socket.SOCK_DGRAM)
        ttl_bin = struct.pack('@i', ttl) # time-to-live, increase to reach beyond local networkt
        if family== socket.AF_INET: # IPv4
            s.send_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl_bin)
        else: # IPv6
            s.send_sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_MULTICAST_HOPS, ttl_bin)

        # udp socket for receving
        (family, type, proto, canonname, s.local_address) = socket.getaddrinfo(*local_address)[0]
        s.recv_sock = socket.socket(family=family, type=socket.SOCK_DGRAM)
        s.recv_sock.settimeout(0)  # timeout immediately
        if ip_address(s.local_address[0]).is_multicast:
            group_bin = socket.inet_pton(family,s.local_address[0]) # multicast address in bytes
            mreq = group_bin + struct.pack('=I', socket.INADDR_ANY)
            if family== socket.AF_INET: # IPv4
                s.recv_sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            else: # IPv6
                s.recv_sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_JOIN_GROUP, mreq)
        s.recv_sock.bind(("",s.local_address[1]))  # Bind socket to local port
        
    def clearRecvBuffer(s):
        """clear old messages that exits in receive socket buffer"""
        try:
            while True:
                _ = s.recv_sock.recv(s.BUFFER_LEN)
        except Exception:
            pass

    def receive(s,
                max_attempts: int = 1000000,
                clear_buf: bool = True,
                newest: bool = True):
        """ return the recived data at local port
            Input:
                max_attempts: int, max num of attempts to receive data
                clear_buff: bool, if True clear prior messages in receive socket
                newest: bool, if True only get the newest data 
        """
        # clear receive socket buffer, any old messages prior to
        # receive will be cleared
        if clear_buf:
            try:
                while True:
                    _ = s.recv_sock.recv(s.BUFFER_LEN)
            except Exception:
                pass
        # try max_attempts times to receive at least one message
        for k in range(max_attempts):
            try:
                recv_data = s.recv_sock.recv(s.BUFFER_LEN)
                break
            except Exception:
                continue
        # only get the newest data if True
        if newest:
            try:
                for k in range(max_attempts):
                    recv_data = s.recv_sock.recv(s.BUFFER_LEN)
            except Exception:
                pass
        try:
            return recv_data
        except UnboundLocalError:
            raise TimeoutError("tried too many times")

    def send(s, data):
        """send the data to remote address, return num_bytes_send"""
        return s.send_sock.sendto(data, s.remote_address)

    def close(s):
        try:
            s.recv_sock.shutdown(socket.SHUT_RDWR)
            s.recv_sock.close()
        except Exception as e:
            print(e)
        print(f"shutdown UDP server:{s.local_address},{s.remote_address}")

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

    