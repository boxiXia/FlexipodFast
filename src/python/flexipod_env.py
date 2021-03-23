import gym
import subprocess
import time
import numpy as np
from collections import defaultdict
import msgpack
import socket
import warnings

class FlexipodEnv(gym.Env):
    """
    openai gym compatible environment for the simulated 12DOF flexipod
    
    ref: https://github.com/openai/gym/blob/master/gym/core.py
    """
    # name of command message
    CMD_NAME = ("header", "t", "cmd")
    CMD_ID = defaultdict(None,{name: k  for k,name in enumerate(CMD_NAME)})
    
    UDP_TERMINATE = int(-1)
    UDP_PAUSE = int(17)
    UDP_RESUME = int(16)
    UDP_RESET = int(15)
    UDP_ROBOT_STATE_REPORT = int(14)
    UDP_MOTOR_VEL_COMMEND = int(13)
    UDP_STEP_MOTOR_VEL_COMMEND = int(12)
    UDP_MOTOR_POS_COMMEND = int(11)
    UDP_STEP_MOTOR_POS_COMMEND = int(10)
    
    def __init__(self, dof = 12, num_observation=5,normalize = True,
           ip_local = "127.0.0.1", port_local = 32000,
           ip_remote = "127.0.0.1",port_remote = 32001):
        
        super(FlexipodEnv,self).__init__()
        self.local_address = (ip_local, port_local)
        self.remote_address = (ip_remote, port_remote)
        self.packer = msgpack.Packer(use_single_float=True, use_bin_type=True)
        
        self.dof = dof # num joints
        self.num_observation = num_observation
        self.normalize = normalize
        
        
        max_joint_vel = 10
        # name of the returned message
        REC_NAME = np.array([
            # name,         size,      min,          max
            ("header",       1,      -32768,         32768        ),
            ("t",            1,      0,              np.inf       ), # simulation time [s]
            ("joint_pos",    dof*2,  -1.,            1.           ), # joint cos(angle) sin(angle) [rad]
            ("joint_vel",    dof,    -max_joint_vel, max_joint_vel), # joint velocity [rad/s]
            ("actuation",    dof,    -1.,            1.           ), # joint actuation [-1,1]
            ("orientation",  6,      -1.,            1.           ), # base link (body) orientation
            ("ang_vel",      3,      -30.,           30.          ), # base link (body) angular velocity [rad/s]
            ("com_acc",      3,      -30.,           30.          ), # base link (body) acceleration
            ("com_vel",      3,      -2.,            2.           ), # base link (body) velocity
            ("spring_strain",128,    -0.1,           0.1          ), # selected spring strain
            ("com_pos",      3.,     -1.,            1.           ), # base link (body) position
        ],dtype=[('name', 'U14'), ('size', 'i4'), ('min', 'f4'),('max', 'f4')])
        
        REC_SIZE = REC_NAME["size"]
        REC_MIN = REC_NAME["min"]
        REC_MAX = REC_NAME["max"]

        
        self.ID =defaultdict(None,{name: k  for k,(name,_,_,_) in enumerate(REC_NAME)})


        OBS_NAME = ("joint_pos","joint_vel","actuation","orientation",
                    "ang_vel","com_acc","com_vel","spring_strain","com_pos")
        # joint_pos,joint_vel,actuation,orientation,ang_vel,com_acc,com_vel,com_pos.z
        state_size = np.sum([REC_NAME[self.ID[name]]["size"] for name in OBS_NAME])-2
        
        # self.max_action = 1.5 # [rad/s]
        self.max_action = 0.05 # [rad], delta position control
                
        self.action_space = gym.spaces.Box(
            low= -self.max_action*np.ones(self.dof,dtype=np.float32),
            high = self.max_action*np.ones(self.dof,dtype=np.float32),
            dtype=np.float32)
        
        self.min_observation = np.hstack([ 
            np.ones(REC_SIZE[self.ID[name]])*REC_MIN[self.ID[name]]for name in OBS_NAME])[:-2]
        self.max_observation = np.hstack([ 
            np.ones(REC_SIZE[self.ID[name]])*REC_MAX[self.ID[name]]for name in OBS_NAME])[:-2]

        self.min_observation = np.tile(self.min_observation,(num_observation,1)).astype(np.float32)
        self.max_observation = np.tile(self.max_observation,(num_observation,1)).astype(np.float32)
        
        self.observation_space = gym.spaces.Box(
            low =self.min_observation,
            high=self.max_observation,
            dtype = np.float32)

        # reset data packed # https://msgpack-python.readthedocs.io/en/latest/api.html#msgpack.Packer
        self.reset_cmd_b = self.packer.pack([self.UDP_RESET,0,[0,0,0,0]])
        self.pause_cmd_b = self.packer.pack([self.UDP_PAUSE,0,[0,0,0,0]])
        self.resume_cmd_b = self.packer.pack([self.UDP_RESUME,0,[0,0,0,0]])
        self.close_cmd_b = self.packer.pack([self.UDP_TERMINATE,0,[0,0,0,0]])
        self.BUFFER_LEN = 32768  # in bytes
        self.TIMEOUT = 0.1 #timeout duration
        
#         self.startSimulation()
#         gc.collect()
        self.send_sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM) # UDP
        self.send_sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1) #immediate reuse of IP address
        
    def __del__(self): # Deleting (Calling destructor) 
        print('Destructor called, FlexipodEnv deleted.')
#         self.close()
    
    def step(self,action = None):
#         step_cmd_b = self.packer.pack([self.UDP_STEP_MOTOR_VEL_COMMEND,time.time(),action])
        if action is not None:
            
            
            # map action -> action*max_acton
            cmd_action = np.multiply(action,self.max_action).astype(np.float32)
            
            # # position difference control
            cmd_action =  cmd_action + self.joint_pos
            step_cmd_b = self.packer.pack([self.UDP_MOTOR_POS_COMMEND,time.time(),cmd_action.tolist()])

            # step_cmd_b = self.packer.pack([self.UDP_MOTOR_VEL_COMMEND,time.time(),cmd_action.tolist()])
#             step_cmd_b = self.packer.pack([self.UDP_STEP_MOTOR_VEL_COMMEND,time.time(),cmd_action])
            num_bytes_send = self.send_sock.sendto(step_cmd_b,self.remote_address)
            
        for k in range(5): # try 5 times
            try:
                msg_rec = self.receive()
                return self._processRecMsg(msg_rec,repeat_first = False)
            except Exception as e: # raise the exception at the last time
                warnings.warn(f"step(): try #{k}:{e}")
        raise TimeoutError("step():tried too many times")

    def _processRecMsg(self,msg_rec,repeat_first = False):
        """processed received message to state action pair"""
        # joint_pos,joint_vel,actuation,orientation,ang_vel,com_acc,com_vel,com_pos.z
#         observation = np.hstack(msg_i[2:-1]+[msg_i[-1][-1]]).astype(np.float32)
        msg_rec_i = msg_rec[0]
        orientation_z = msg_rec_i[self.ID['orientation']][2]
        actuation = msg_rec_i[self.ID['actuation']][2]
        # com_vel = np.linalg.norm(msg_rec_i[self.ID['com_vel']])
        com_z = msg_rec_i[self.ID['com_pos']][2]
        
        joint_pos = msg_rec_i[self.ID['joint_pos']]
        self.joint_pos = np.arctan2(joint_pos[1::2],joint_pos[::2]).astype(np.float32) # convert to rad
        # print(self.joint_pos)
        
        id_com_pos = self.ID['com_pos']
        if repeat_first:
             observation = np.stack(
                [np.hstack(msg_i[2:-1]+[com_z]).astype(np.float32) for 
                msg_i in [msg_rec[0]]*self.num_observation] )
        else: 
            observation = np.stack(
                [np.hstack(msg_i[2:-1]+[msg_i[id_com_pos][2]]).astype(np.float32) 
                 for msg_i in msg_rec] )
            
        observation = (observation/self.max_observation)

#         print(orientation_z,com_z)
        # reward = orientation_z-0.8 + (com_z-0.3)-0.2*min(1.0,com_vel)
        uph_cost = orientation_z + com_z-0.15
        quad_ctrl_cost  = 0.1 * np.square(actuation).sum() # quad control cost
        reward =  uph_cost - quad_ctrl_cost
        
#         reward = orientation_z
        done = True if (orientation_z<0.6)or(com_z<0.2) else False
        t = msg_rec_i[self.ID['t']]
        info = {'t':t}
        # if done:
#             print(orientation_z,com_z)
        return observation,reward,done,info
    
    def reset(self):
        for k in range(5):# try 5 times
            try:
                self.send_sock.sendto(self.reset_cmd_b,self.remote_address)
                time.sleep(1/400)
                msg_rec = self.receive()
                observation,reward,done,info =  self._processRecMsg(msg_rec,repeat_first = True)
                self.episode_start_time = info['t']
                return observation
            except Exception as e:
                warnings.warn(f"reset(): try #{k}:{e}")
        raise TimeoutError("step():tried too many times")
    
    def render(self,mode="human"):
        pass

    def startSimulation(self):
        try: # check if the simulation is opened
            msg_rec = self.receive()
        except socket.timeout: # program not opened
            task = subprocess.Popen(["run_flexipod.bat"])
            
    def endSimulation(self):
            self.send_sock.sendto(self.close_cmd_b,self.remote_address)

    def receive(self,reset=False):
        try:
            sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM) # UDP
#             sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1) #immediate reuse of IP address
            sock.settimeout(self.TIMEOUT)
            sock.bind(self.local_address) #Bind the socket to the port
            
            data = sock.recv(self.BUFFER_LEN)
            msg_rec = msgpack.unpackb(data)
#             print(len(data))
             # closing connection
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()
            
            data_unpacked = msgpack.unpackb(data)
            return data_unpacked
        except Exception as e:
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()
            raise e
        
    def item_to_String(key,value):
        try:
            return f"{key:<12s}:{','.join(f'{k:>+6.2f}' for k in value)}\n"
        except:
            return f"{key:<12s}:{value:>+6.2f}\n"
        
    def pause(self):
        self.send_sock.sendto(self.pause_cmd_b,self.remote_address)
    def resume(self):
        self.send_sock.sendto(self.resume_cmd_b,self.remote_address)




if __name__ == '__main__':
    env = FlexipodEnv(dof = 12)