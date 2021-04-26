import gym
import subprocess
import time
import numpy as np
from collections import defaultdict
import msgpack
import socket
import warnings

def linearMinMaxConversionCoefficient(a,b,c,d):
    """
    linear conversion from [a,b] to [c,d] using y = kx + m
    return coefficient k,m
    """
    k = (d - c)/(b-a)
    m = c - a*k
    return k,m

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
        
        self.dof = dof # num joints
        self.num_observation = num_observation
        self.normalize = normalize
        
        self._max_episode_steps = np.inf
        
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
            ("spring_strain",128,   -0.005,         0.005         ), # selected spring strain
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
        
        # action min-max
        max_action = 0.02 # [rad], delta position control
        # raw min/max action
        self.raw_min_act = - max_action*np.ones(self.dof,dtype=np.float32) # [rad], delta position control for all motors
        self.raw_max_act =   max_action*np.ones(self.dof,dtype=np.float32) # [rad], delta position control for all motors
        
        # observartion min-max
        self.raw_min_obs = np.hstack([ # raw max observation (from simulation)
            np.ones(REC_SIZE[self.ID[name]])*REC_MIN[self.ID[name]]for name in OBS_NAME]).astype(np.float32)[:-2]
        self.raw_max_obs = np.hstack([ # raw min observation (from simulation)
            np.ones(REC_SIZE[self.ID[name]])*REC_MAX[self.ID[name]]for name in OBS_NAME]).astype(np.float32)[:-2]

        # multiple observation per network input
        self.raw_min_obs = np.tile(self.raw_min_obs,(num_observation,1))
        self.raw_max_obs = np.tile(self.raw_max_obs,(num_observation,1))
        
        if normalize: # conditionally normalize the action space
            self.action_space = gym.spaces.Box(low = - np.ones_like(self.raw_min_act),
                                               high = np.ones_like(self.raw_max_act))
            self.observation_space = gym.spaces.Box(low = - np.ones_like(self.raw_min_obs),
                                                    high = np.ones_like(self.raw_max_obs))
            # action conversion from normalized to raw: y = kx + m
            self.to_raw_act_k,self.to_raw_act_m = linearMinMaxConversionCoefficient(
                a = -1.0,b = 1.0,c = self.raw_min_act,d = self.raw_max_act)
            # action conversion from raw to normalized: nor - normalized
            self.to_nor_act_k, self.to_nor_act_m = linearMinMaxConversionCoefficient(
                a = self.raw_min_act,b = self.raw_max_act, c = -1.0,d = 1.0)
            # observation from normalized to raw:
            self.to_raw_obs_k,self.to_raw_obs_m = linearMinMaxConversionCoefficient(
                a = -1.0, b = 1.0, c = self.raw_min_obs, d = self.raw_max_obs)
            # observation from raw to normalized:
            self.to_nor_obs_k, self.to_nor_obs_m = linearMinMaxConversionCoefficient(
                a = self.raw_min_obs,b = self.raw_max_obs, c = -1.0,d = 1.0)
        else: # raw action and observation
            self.action_space = gym.spaces.Box(low = self.raw_min_act,high = self.raw_max_act)
            self.observation_space = gym.spaces.Box(low = self.raw_min_obs,high = self.raw_max_obs)

        # reset data packed # https://msgpack-python.readthedocs.io/en/latest/api.html#msgpack.Packer
        self.packer = msgpack.Packer(use_single_float=True, use_bin_type=True)
        # self.reset_cmd_b = self.packer.pack([self.UDP_RESET,0,[0,0,0,0]])
        self.reset_cmd_b = self.packer.pack([self.UDP_RESET,0])

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
            cmd_action = np.asarray(action,dtype=np.float32)
            if self.normalize:
                cmd_action = cmd_action*self.to_raw_act_k + self.to_raw_act_m            
            
            # # position difference control
            cmd_action += self.joint_pos
            step_cmd_b = self.packer.pack([self.UDP_MOTOR_POS_COMMEND,time.time(),cmd_action.tolist()])

            # step_cmd_b = self.packer.pack([self.UDP_MOTOR_VEL_COMMEND,time.time(),cmd_action.tolist()])
#             step_cmd_b = self.packer.pack([self.UDP_STEP_MOTOR_VEL_COMMEND,time.time(),cmd_action])
            num_bytes_send = self.send_sock.sendto(step_cmd_b,self.remote_address)
        msg_rec = self.receive()
        return self._processRecMsg(msg_rec)

    def _processRecMsg(self,msg_rec):
        """processed received message to state action pair"""
        # joint_pos,joint_vel,actuation,orientation,ang_vel,com_acc,com_vel,com_pos.z
#         observation = np.hstack(msg_i[2:-1]+[msg_i[-1][-1]]).astype(np.float32)
        msg_rec_i = msg_rec[0]
        orientation_z = msg_rec_i[self.ID['orientation']][2]
        actuation = msg_rec_i[self.ID['actuation']] # actuation (size=dof) of the latest observation
        # com_vel = np.linalg.norm(msg_rec_i[self.ID['com_vel']])
        com_z = msg_rec_i[self.ID['com_pos']][2]
        
        joint_pos = msg_rec_i[self.ID['joint_pos']]
        self.joint_pos = np.arctan2(joint_pos[1::2],joint_pos[::2]).astype(np.float32) # convert to rad
        # print(self.joint_pos)
        id_com_pos = self.ID['com_pos']

        observation = np.stack(
            [np.hstack(msg_i[2:-1]+[msg_i[id_com_pos][2]]).astype(np.float32) 
                for msg_i in msg_rec] )
        if self.normalize: # normalize the observation
            observation = observation*self.to_nor_obs_k + self.to_nor_obs_m

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
        # _ = [m[self.ID['t']] for m in msg_rec]
        # print(_[0]-_[-1])
        return observation,reward,done,info # TODO, change shape
    
    def reset(self):
        self.send_sock.sendto(self.reset_cmd_b,self.remote_address)
        time.sleep(1/100)
        msg_rec = self.receive()
        observation,reward,done,info =  self._processRecMsg(msg_rec)
        self.episode_start_time = info['t']
        # print([m[self.ID['t']] for m in msg_rec])
        return observation
    
    def render(self,mode="human"):
        pass

    def startSimulation(self):
        try: # check if the simulation is opened
            msg_rec = self.receive()
        except socket.timeout: # program not opened
            task = subprocess.Popen(["run_flexipod.bat"])
            
    def endSimulation(self):
            self.send_sock.sendto(self.close_cmd_b,self.remote_address)

    def receive(self,max_attempts:int = 10):
        try:
            sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM) # UDP
            # sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1) #immediate reuse of IP address
            sock.settimeout(self.TIMEOUT)
            sock.bind(self.local_address) #Bind the socket to the port
            for k in range(max_attempts):# try max_attempts times
                try:
                    data = sock.recv(self.BUFFER_LEN)
                    sock.shutdown(socket.SHUT_RDWR) # closing connection
                    sock.close()
                    msg_rec = msgpack.unpackb(data)
                    data_unpacked = msgpack.unpackb(data)
                    return data_unpacked
                except Exception as e:
                    print(f"receive(): try #{k}:{e}")
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()
            raise TimeoutError("receive():tried too many times")
        except KeyboardInterrupt:
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()
            raise KeyboardInterrupt
    
    @staticmethod   
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