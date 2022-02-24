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
from flexipod_env import FlexipodEnv
from pulley_control import PulleyControl
from udp_server import UDPServer

class SoftHumanoidServer:
    def __init__(s,
                 local_address = ("192.168.137.1",32005),
                 remote_address = ("192.168.137.130",32004)
                 ):
        s.timeout = 2 # timeout in [second]
        s.server = UDPServer(local_address=local_address,remote_address=remote_address)
        # reset data packed # https://msgpack-python.readthedocs.io/en/latest/api.html#msgpack.Packer
        s.packer = msgpack.Packer(use_single_float=True, use_bin_type=True)
        
        s.HEADER_POS_CONTROL = 10001

        s.ID_t = 0
        s.ID_joint_pos = 1
        s.ID_joint_vel = 2
        s.ID_joint_torque = 3
        s.ID_orientation = 4
        # s.ID_com_vel
        s.ID_com_acc =  6
        # s.ID_com_pos
        s.ID_ang_vel = 5 # TODO: check its oreientation
        s.ID_constraint = 8

        s._set_reset_pos() # set the joint reset position

    def _set_reset_pos(s):
        """ set the reset position"""
        pi = np.pi

        p_f_b0 = -pi/6 # font arm body-0
        p_f_01 = pi/3 # front arm 01
        p_f_12 = -pi/3 # front arm 12

        p_bl_b0 = pi/12
        p_br_b0 = pi/12

        p_bl_01 = -p_bl_b0*2
        p_br_01 = -p_br_b0*2

        p_bl_12 = p_bl_b0
        p_br_12 = p_br_b0

        # n0 = pi/20 # body incline
        n0 = 0 # body incline

        s.joint_pos_reset = np.array([
            # front left
            -p_f_b0,      # 0
            -p_f_01,      # 1
            -p_f_12,      # 2
            # front right
            p_f_b0,      # 3
            p_f_01,      # 4
            p_f_12,      # 5
            # back left
            -p_bl_b0-n0,   # 6
            -p_bl_01,      # 7
            -p_bl_12,      # 8
            #back right
            p_br_b0+n0,   # 9
            p_br_01,      # 10
            p_br_12,      # 11
        ])

        s.joint_pos_reset = np.zeros(12)
            
    def receive(s,verbose=False):
        data = s.server.receive(timeout=s.timeout,clear_buf=True,newest=False) # max of timeout second
        # print(len(data))
        data_unpacked = msgpack.unpackb(data,use_list=False)
        joint_pos = data_unpacked[1][0][s.ID_joint_pos]
        s.joint_pos = np.arctan2(joint_pos[1::2],joint_pos[::2],dtype=np.float32)
        return data_unpacked
    
    def send(s,action):
        packed = s.packer.pack([s.HEADER_POS_CONTROL, action.tolist() if type(action) is np.ndarray else action])
        return s.server.send(packed)

    def reset(s):
        s.send(s.joint_pos_reset)
        s.receive()
        while(np.linalg.norm(s.joint_pos-s.joint_pos_reset,ord=1)>0.2):
            s.send(s.joint_pos_reset)
            s.receive()

class PhysicalHumanoid(FlexipodEnv):
    def __init__(
        s,
        local_address = ("127.0.0.1",33300),
        remote_address = ("127.0.0.1",33301),
        robot_folder:str = "../../../robot/v11"
        ):
        super().__init__() # init gym env

    def _implInitServer(s):
        """implementation specific initiailzation of the server"""
        # defualt packer # https://msgpack-python.readthedocs.io/en/latest/api.html#msgpack.Packer
        s.packer = msgpack.Packer(use_single_float=True, use_bin_type=True)
        # s.pulley = PulleyControl()
        s.robot = SoftHumanoidServer()

    def _impl_SendAction(s,cmd_action:list):
        """
        implementation specific function to serialize and send the action 
        processed by act() to the env.
        """
        # send to robot
        num_bytes_send = s.robot.send(cmd_action)
        # send to other peripherials
        # TODO

    def _impl_Receive(s,timeout:float = 4):
        """Implementation specific receive messages from the env"""
        data_unpacked = s.robot.receive()
        return data_unpacked 


        
    def start(s):
        """start the environment"""
        # s.pulley.reset() 
        s.robot.reset()
        print(s.robot.joint_pos)


if __name__ == '__main__':
    bot = PhysicalHumanoid()
    bot.start()
    bot._impl_Receive()
