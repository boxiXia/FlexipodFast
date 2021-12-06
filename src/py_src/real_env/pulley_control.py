import msgpack
import numpy as np
import time
from typing import Union

# fix relative import
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# import from parent directory
from udp_server import UDPServer


class PulleyControl:
    def __init__(self,
                 local_address = ("192.168.137.1",32003),
                 remote_address = ("192.168.137.123",32002),
                 timeout=2 # [second]
                 ):
        self.server = UDPServer(local_address=local_address,remote_address=remote_address)
        # reset data packed # https://msgpack-python.readthedocs.io/en/latest/api.html#msgpack.Packer
        self.packer = msgpack.Packer(use_single_float=True, use_bin_type=True)
        self.reset_pos = 0
        self.timeout=timeout
        self.setResetPos()
#         self.reset()
        
    def setResetPos(self, reset_pos:Union[float,int,str] = "auto"):
        """set the pully reset position
            input can be a float position [deg] or "auto"
        """
        if type(reset_pos)!=str and np.isscalar(reset_pos):
            self.reset_pos = float(reset_pos)
        elif reset_pos == "auto":
            self.receive()
            # place the robot centered and upright
            # release the pully until force is below some threashold
            dx = 0
            k = 1
            thresh = 0.1
            while self.cur>thresh: 
                self.receive()
                self.send(self.pos+dx,k) # TODO: change API
                dx-=0.1
                
            # pull the pully until force is above some threashold
            dx = 0
            k = 10
            thresh = 0.1
            while self.cur<thresh:
                self.receive()
                self.send(self.pos+dx,k) # TODO: change API
                dx+=0.01
            self.reset_pos = self.pos
        else:
            raise NotImplementedError("reset_pos should be either float, int or 'auto'")
        
    

    def _resetLoop(self,k:float,dx:float,thresh:float):
        """helper funtion to first pull up and then down
            Args:
                k: motor stiffness coefficient, higher k -> stiffer pully
                dx: float [deg] to pull up above the reset position
                thresh: float [0-1], threshold for stopping the loop
        """
#         self.send(self.reset_pos+dx, k) # move up
#         self.send(self.reset_pos+dx, k) # move up
#         time.sleep(1) # wait until stable
#         self.receive()
        dx_orginal = dx
        while self.pos < self.reset_pos+dx_orginal:
            self.send(self.reset_pos+dx, k) # move up
            dx+=0.1
            self.receive()
        time.sleep(1)
        while self.cur>thresh: # move down until current is below a threshold
            self.send(self.reset_pos+dx,k) # TODO: change API
            dx-=0.1
            self.receive()
        
        
    def reset(self):
        """pull the robot to the reset position"""
        self._resetLoop(k=10,dx=60,thresh=0.1) # loop#1
        self._resetLoop(k=10,dx=60,thresh=0.1) # loop#2
        self._resetLoop(k=10,dx=40,thresh=0.1) # loop#2
        self.send(self.reset_pos, 0.1) # set to reset position
        
    def receive(self,verbose=False):
        data = self.server.receive(timeout=self.timeout,clear_buf=True,newest=False) # max of timeout second
        # print(len(data))
        data_unpacked = msgpack.unpackb(data,use_list=False)
        #time[s], position[deg], velocity[deg/s], current [-1,1]
        self.t, (self.pos,), (self.vel,), (self.cur,) = data_unpacked
#         print(data_unpacked)

    def send(self,action:float,k):
        self.server.send(self.packer.pack([[action],[k]]))

    # def reset(self):
    def move(self,dx,k=0.5):
        self.receive()
        self.send(self.pos +dx,k)

if __name__ == '__main__':
    pulley = PulleyControl()
    # pulley.move(-600,10)
    pulley.reset()