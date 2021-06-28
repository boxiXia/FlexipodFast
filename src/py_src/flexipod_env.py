import gym
import subprocess
import time
import numpy as np
from collections import defaultdict
import msgpack
import socket
import os

import pybullet as p
import pybullet_utils.bullet_client as bc

def linearMinMaxConversionCoefficient(a,b,c,d):
    """
    linear conversion from [a,b] to [c,d] using y = kx + m
    return coefficient k,m
    """
    k = (d - c)/(b-a)
    m = c - a*k
    return k,m

class BulletCollisionDetect:
    """
    Helper class for collision detection using pybullet
    """
    def __init__(self,gui=False):
        """
        load urdf and init the collision detection
        """
        urdf_path="../../data/urdf/12dof/robot.urdf"
        urdf_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),urdf_path)
        # print(os.path.abspath(urdf_path))
        # loading using bullet_client
        # ref: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#
        self._p = bc.BulletClient(p.GUI if gui else p.DIRECT)
        self.body_id = self._p.loadURDF(urdf_path,
                              useFixedBase=1,
                              flags=p.URDF_USE_SELF_COLLISION)  # load urdf
        self.dof = self._p.getNumJoints(self.body_id)
        # JOINT_INFO_DICT = {name: i for i, name in enumerate([
        #  "jointIndex", "jointName", "jointType", "qIndex", "uIndex",
        #  "flags", "jointDamping", "jointFriction", "jointLowerLimit",
        #  "jointUpperLimit", "jointMaxForce", "jointMaxVelocity", "linkName",
        #  "jointAxis", "parentFramePos", "parentFrameOrn", "parentIndex"])}
#         joint_info = [self._p.getJointInfo(self.body_id, i) for i in range(self.dof)]
#         self.joint_lower_limit = np.asarray(
#             [j[JOINT_INFO_DICT["jointLowerLimit"]] for j in joint_info])
#         self.joint_upper_limit = np.asarray(
#             [j[JOINT_INFO_DICT["jointUpperLimit"]] for j in joint_info])
#         self.joint_range = self.joint_upper_limit-self.joint_lower_limit

    def getContactPoints(self,joint_pos):
        """
        get contact points info given joint position in rad
        ref:https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.cb0co8y2vuvc
        """
        for i in range(self.dof):
            self._p.resetJointState(self.body_id, i, joint_pos[i],0)
        self._p.performCollisionDetection()
        return self._p.getContactPoints()
        
    def __del__(self):
        self._p.disconnect()
#         print("deleted")

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
    
    def __init__(
        self, 
        dof = 12, 
        num_observation=5,
        num_sensors = 0, # num of spring strain sensors
        normalize = True,
        max_joint_vel = 10, # maximum joint velocity rad/s
        humanoid_task = True, # if true, humanoid, else qurdurped
        ip_local = "127.0.0.1", port_local = 33300,
        ip_remote = "127.0.0.1",port_remote = 33301):
        
        super(FlexipodEnv,self).__init__()
        
        self.humanoid_task = humanoid_task
        
        print(f"humanoid_task:{humanoid_task}")
        self.cd = BulletCollisionDetect() # collision detection with pybullet
        
        self.local_address = (ip_local, port_local)
        self.remote_address = (ip_remote, port_remote)
        
        self.dof = dof # num joints
        self.joint_pos = np.empty(dof,dtype=np.float32) # joint position [rad]
        self.num_observation = num_observation
        self.normalize = normalize
        
        self._max_episode_steps = 10000#np.inf
        self.episode_steps = 0 # curret step in an episode

        self.info = False # bool flag to send info or not when processing messages
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
            ("spring_strain",num_sensors,-0.005,    0.005         ), # selected spring strain
            ("com_pos",      3.,     -1.,            1.           ), # base link (body) position
        ],dtype=[('name', 'U14'), ('size', 'i4'), ('min', 'f4'),('max', 'f4')])
        
        REC_SIZE = REC_NAME["size"]
        REC_MIN = REC_NAME["min"]
        REC_MAX = REC_NAME["max"]

        
        self.ID =defaultdict(None,{name: k  for k,(name,_,_,_) in enumerate(REC_NAME)})
        
        self.ID_t = self.ID['t']
        self.ID_actuation = self.ID['actuation']
        self.ID_orientation = self.ID['orientation']
        self.ID_joint_pos = self.ID['joint_pos']
        self.ID_com_pos = self.ID['com_pos']
        self.ID_com_vel = self.ID['com_vel']
        
        OBS_NAME = ("joint_pos","joint_vel","actuation","orientation",
                    "ang_vel","com_acc","com_vel","spring_strain")#,"com_pos")
        # joint_pos,joint_vel,actuation,orientation,ang_vel,com_acc,com_vel,com_pos.z
        
        # action min-max
        # max_action = 0.025 # [rad], delta position control
        max_action = 5 # [rad/s], velocity control

        # raw min/max action
        self.raw_min_act = - max_action*np.ones(self.dof,dtype=np.float32) # [rad], delta position control for all motors
        self.raw_max_act =   max_action*np.ones(self.dof,dtype=np.float32) # [rad], delta position control for all motors
        
        # observartion min-max
        self.raw_min_obs = np.hstack([ # raw max observation (from simulation)
            np.ones(REC_SIZE[self.ID[name]])*REC_MIN[self.ID[name]]for name in OBS_NAME]).astype(np.float32)#[:-2]
        self.raw_max_obs = np.hstack([ # raw min observation (from simulation)
            np.ones(REC_SIZE[self.ID[name]])*REC_MAX[self.ID[name]]for name in OBS_NAME]).astype(np.float32)#[:-2]

        # multiple observation per network input
        self.raw_min_obs = np.tile(self.raw_min_obs,(num_observation,1))
        self.raw_max_obs = np.tile(self.raw_max_obs,(num_observation,1))
        
        self.flatten_obs = True # whether to flatten the observation
        if(self.flatten_obs):
            self.raw_min_obs = self.raw_min_obs.ravel()
            self.raw_max_obs = self.raw_max_obs.ravel()
        
        if normalize: # conditionally normalize the action space
            self.action_space = gym.spaces.Box(low = - np.ones_like(self.raw_min_act),high = np.ones_like(self.raw_max_act))
            self.observation_space = gym.spaces.Box(low = - np.ones_like(self.raw_min_obs),high = np.ones_like(self.raw_max_obs))
            # action conversion from normalized to raw: y = kx + m
            self.to_raw_act_k,self.to_raw_act_m = linearMinMaxConversionCoefficient(-1.0,1.0,self.raw_min_act,self.raw_max_act)
            # action conversion from raw to normalized: nor - normalized
            self.to_nor_act_k, self.to_nor_act_m = linearMinMaxConversionCoefficient(self.raw_min_act,self.raw_max_act,-1.0,1.0)
            # observation from normalized to raw:
            self.to_raw_obs_k,self.to_raw_obs_m = linearMinMaxConversionCoefficient(-1.0, 1.0,self.raw_min_obs,self.raw_max_obs)
            # observation from raw to normalized:
            self.to_nor_obs_k, self.to_nor_obs_m = linearMinMaxConversionCoefficient(self.raw_min_obs,self.raw_max_obs,-1.0,1.0)
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
        self.TIMEOUT = 0.2 #timeout duration
        
#         self.startSimulation()
#         gc.collect()
        self.send_sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM) # UDP
        self.send_sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1) #immediate reuse of IP address
        
        # start the simulation
        self.startSimulation()
        
    def __del__(self): # Deleting (Calling destructor) 
        print('Destructor called, FlexipodEnv deleted.')
#         self.close()
    
    def step(self,action = None):
        if action is not None:
            # map action -> action*max_acton
            cmd_action = np.asarray(action,dtype=np.float32)
            if self.normalize:
                cmd_action = cmd_action*self.to_raw_act_k + self.to_raw_act_m
            # # # position difference control
            # cmd_action += self.joint_pos # the actual position
            
            # if len(self.cd.getContactPoints(cmd_action))>0:
            # step_cmd_b = self.packer.pack([self.UDP_MOTOR_POS_COMMEND,time.time(),cmd_action.tolist()])
            step_cmd_b = self.packer.pack([self.UDP_MOTOR_VEL_COMMEND,time.time(),cmd_action.tolist()])
            num_bytes_send = self.send_sock.sendto(step_cmd_b,self.remote_address)
        msg_rec = self.receive()
        self.episode_steps+=1 # update episodic step counter
        return self._processRecMsg(msg_rec)

    def _processRecMsg(self,msg_rec):
        """processed received message to state action pair"""
        # joint_pos,joint_vel,actuation,orientation,ang_vel,com_acc,com_vel,com_pos.z
        # observation = np.hstack(msg_i[2:-1]+[msg_i[-1][-1]]).astype(np.float32)
        
        msg_rec_i = msg_rec[0]
        
        actuation = msg_rec_i[self.ID_actuation] # actuation (size=dof) of the latest observation
        com_z = msg_rec_i[self.ID_com_pos][2]
        # joint position (sin,cos->rad)
        joint_pos = msg_rec_i[self.ID_joint_pos]
        _ = np.arctan2(joint_pos[1::2],joint_pos[::2],self.joint_pos,dtype=np.float32)
        # print(f"self.joint_pos ={self.joint_pos}")
        
        
        # observation = np.stack([np.hstack(msg_i[2:-1]+[msg_i[self.ID_com_pos][2]])
        observation = np.stack([np.hstack(msg_i[2:-1])
                for msg_i in msg_rec]).astype(np.float32) 
        
        if self.flatten_obs:
            observation = observation.ravel()
        if self.normalize: # normalize the observation
            observation = observation*self.to_nor_obs_k + self.to_nor_obs_m
        
        # x velocity
        com_vel_x = sum([msg_i[self.ID_com_vel][0] for msg_i in msg_rec])/len(msg_rec)
        # vel_cost = 0.3*np.clip(com_vel_xy,0,1)+0.7
        vel_cost = 0.4*np.clip(com_vel_x,-0.5,1)+0.6
        
#         print(orientation_z,com_z)
        # uph_cost = max(0,orientation_z)*min(com_z+0.56,1)
        ori = msg_rec_i[self.ID_orientation]
        
        if self.humanoid_task:
            orientation_z = ori[2] # z_z, local x vector projected to world z direction
            uph_cost = (np.clip(orientation_z*1.02,0,1)**3)*min(com_z+0.56,1)
            com_z_min = 0.36
        else:
            orientation_z= ori[0]*ori[4] - ori[1]*ori[3] # z_z, local z vector projected to world z direction
            uph_cost = (np.clip(orientation_z*1.02,0,1)**3)*min(com_z+0.56,1)
            com_z_min = 0.2
        
        # x = np.linspace(0,1,400)
        # y = np.clip(np.cos(x*np.pi/2)/np.cos(np.pi/180*15),-1,1)**3
        # plt.plot(x,y)
        # print(com_z+0.56)
        # print(msg_rec_i[self.ID_orientation])
        
        quad_ctrl_cost = max(0,1-0.1 * sum(np.square(actuation))) # quad control cost
        reward =  uph_cost*quad_ctrl_cost*vel_cost
        
#         reward = orientation_z
        done = True if (orientation_z<0.65)or(com_z<com_z_min)or(self.episode_steps>=self._max_episode_steps) else False
        # done = True if self.episode_steps>=self._max_episode_steps else False # done when exceeding max steps
        
        t = msg_rec_i[self.ID_t]
        if self.info:
            info = {'t':t,'vel_cost':vel_cost,'uph_cost':uph_cost,'quad_ctrl_cost':quad_ctrl_cost}
        else:
            info = {'t':t}
        # if done:
#             print(orientation_z,com_z)
        # _ = [m[self.ID['t']] for m in msg_rec]
        # print(_[0]-_[-1])
        return observation,reward,done,info # TODO, change shape
    
    def reset(self):
        self.send_sock.sendto(self.reset_cmd_b,self.remote_address)
        time.sleep(1/50)
        msg_rec = self.receive()
        self.episode_steps = 0
        observation,reward,done,info =  self._processRecMsg(msg_rec)
        self.episode_start_time = info['t']
        # print([m[self.ID['t']] for m in msg_rec])
        return observation
    
    def render(self,mode="human"):
        pass

    def startSimulation(self):
        try: # check if the simulation is opened
            msg_rec = self.receive(max_attempts=2,verbose=False)
        except TimeoutError: # program not opened
            # path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"./run_flexipod.bat")
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..\\..\\build\\flexipod.exe")
            # print(path)
            task = subprocess.Popen([path])
            
    def endSimulation(self):
            self.send_sock.sendto(self.close_cmd_b,self.remote_address)

    def receive(self,max_attempts:int = 10,verbose=True):
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
                    if verbose:
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

def make(**kargs):
    return FlexipodEnv(**kargs)
    # print(kargs)
    # return 0

if __name__ == '__main__':
    env = FlexipodEnv(dof = 12)
    env.reset()
    while True:
        action = env.action_space.sample()
        env.step(action)
        # time.sleep(0.0001)
    # print(subprocess.Popen(["cmd"], shell=True))
    # print(subprocess.Popen(["set CUDA_VISIBLE_DEVICES=0"], shell=True))

    # env.startSimulation()
    # print("exit python")