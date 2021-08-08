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
    linear conversion from [a,b] to [c,d] using y = kx + m,
    return coefficient k,m
    """
    k = (d - c)/(b-a)
    m = c - a*k
    return k,m

def linearMapFcn(a,b,c,d):
    """
    return a function of the linear conversion from 
    [a,b] to [c,d] using y = kx + m,
    """
    k = (d - c)/(b-a)
    m = c - a*k    
    def y(x):
        return k*x +m
    return y


class UDPServer:
    def __init__(
        s, # self
        local_address = ("127.0.0.1",33300),
        remote_address = ("127.0.0.1",33301)
        ):        
        s.local_address = local_address
        s.remote_address = remote_address
        s.BUFFER_LEN = 32768  # in bytes
                
        # udp socket for sending
        s.send_sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM) 
        s.send_sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1) 
                
        # udp socket for receving
        s.recv_sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        s.recv_sock.settimeout(0) # timeout immediately
        s.recv_sock.setblocking(False)
        s.recv_sock.bind(s.local_address) #Bind the socket to the port
        
    def receive(s,max_attempts=1000000):
        """return the recived data at local port"""
        for k in range(max_attempts):
            try:
                data = s.recv_sock.recv(s.BUFFER_LEN)
                break
            except Exception:
                continue
        try:
            for k in range(max_attempts):
                data = s.recv_sock.recv(s.BUFFER_LEN)
        except:
            try:
                return data
            except UnboundLocalError:
                raise TimeoutError("tried to many times")
        
    def send(s,data):
        """send the data to remote address, return num_bytes_send"""
        return s.send_sock.sendto(data,s.remote_address) 
        
    def close(s):
        try:
            s.recv_sock.shutdown(socket.SHUT_RDWR)
            s.recv_sock.close()
        except Exception as e:
            print(e)
        print(f"shutdown UDP server:{s.local_address},{s.remote_address}")
    def __del__(s): 
        s.close()


class BulletCollisionDetect:
    """
    Helper class for collision detection using pybullet
    """
    def __init__(self,gui=False,urdf_path="../../data/urdf/12dof/robot.urdf"):
        """
        load urdf and init the collision detection
        """
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
        max_episode_steps = 10000,
        max_action = 5, # [rad/s], velocity control
        local_address = ("127.0.0.1",33300),
        remote_address = ("127.0.0.1",33301)):
        
        super().__init__() # init gym env
        
        self.humanoid_task = humanoid_task
        
        # defualt packer
        # reset data packed # https://msgpack-python.readthedocs.io/en/latest/api.html#msgpack.Packer
        self.packer = msgpack.Packer(use_single_float=True, use_bin_type=True)
        
        # collision detection with pybullet
        self.cd = BulletCollisionDetect() 
        
        
        self.dof = dof # num joints
        self.joint_pos = np.empty(dof,dtype=np.float32) # joint position [rad]
        self.num_observation = num_observation
        self.normalize = normalize
        
        self._max_episode_steps = max_episode_steps # maximum episode steps
        self.episode_steps = 0 # curret step in an episode

        self.info = False # bool flag to send info or not when processing messages
        # name of the returned message
        REC = np.array([
            # name,         size,         min,          max
            ("header",       1,         -32768,         32768        ),
            ("t",            1,         0,              np.inf       ), # simulation time [s]
            ("joint_pos",    dof*2,     -1.,            1.           ), # joint cos(angle) sin(angle) [rad]
            ("joint_vel",    dof,       -max_joint_vel, max_joint_vel), # joint velocity [rad/s]
            ("actuation",    dof,       -1.,            1.           ), # joint actuation [-1,1]
            ("orientation",  6,         -1.,            1.           ), # base link (body) orientation
            ("ang_vel",      3,         -30.,           30.          ), # base link (body) angular velocity [rad/s]
            ("com_acc",      3,         -30.,           30.          ), # base link (body) acceleration
            ("com_vel",      3,         -2.,            2.           ), # base link (body) velocity
            ("spring_strain",num_sensors,-0.005,       0.005         ), # selected spring strain
            ("com_pos",      3.,        -1.,            1.           ), # base link (body) position
        ],dtype=[('name', 'U20'), ('size', 'i4'), ('min', 'f4'),('max', 'f4')])
        

        self.ID =defaultdict(None,{name: k  for k,name in enumerate(REC["name"])})
        
        self.ID_t = self.ID['t']
        self.ID_actuation = self.ID['actuation']
        self.ID_orientation = self.ID['orientation']
        self.ID_joint_pos = self.ID['joint_pos']
        self.ID_com_pos = self.ID['com_pos']
        self.ID_com_vel = self.ID['com_vel']
        self.ID_com_acc = self.ID['com_acc']
        
        OBS_NAME = ("joint_pos","joint_vel","actuation","orientation",
                    "ang_vel","com_acc","com_vel","spring_strain")#,"com_pos")        

        self._initObsACt(max_action,REC,OBS_NAME) # initialize the observation and action space

        # simulation specific commend
        self.reset_cmd_b =  self.packer.pack([self.UDP_RESET,0])
        self.pause_cmd_b =  self.packer.pack([self.UDP_PAUSE,0,[0,0,0,0]])
        self.resume_cmd_b = self.packer.pack([self.UDP_RESUME,0,[0,0,0,0]])
        self.close_cmd_b =  self.packer.pack([self.UDP_TERMINATE,0,[0,0,0,0]])
        
        self.server = UDPServer(local_address,remote_address)

        # start the simulation
        self.start()
        
        
    def _initObsACt(self,max_action,REC,OBS_NAME):
        """helper function to initialize the observation and action space"""
        # raw min/max action for all motors # [rad/s]
        self.raw_min_act = - max_action*np.ones(self.dof,dtype=np.float32)
        self.raw_max_act =   max_action*np.ones(self.dof,dtype=np.float32)
        
        # observartion min-max
        self.raw_min_obs = np.hstack([ # raw max observation (from simulation)
            np.ones(REC["size"][self.ID[name]])*REC["min"][self.ID[name]]for name in OBS_NAME]).astype(np.float32)#[:-2]
        self.raw_max_obs = np.hstack([ # raw min observation (from simulation)
            np.ones(REC["size"][self.ID[name]])*REC["max"][self.ID[name]]for name in OBS_NAME]).astype(np.float32)#[:-2]

        # multiple observation per network input
        self.raw_min_obs = np.tile(self.raw_min_obs,(self.num_observation,1))
        self.raw_max_obs = np.tile(self.raw_max_obs,(self.num_observation,1))
        
        self.flatten_obs = True # whether to flatten the observation
        if(self.flatten_obs):
            self.raw_min_obs = self.raw_min_obs.ravel()
            self.raw_max_obs = self.raw_max_obs.ravel()
        
        if self.normalize: # conditionally normalize the action space
            self.action_space = gym.spaces.Box(low = - np.ones_like(self.raw_min_act),high = np.ones_like(self.raw_max_act))
            self.observation_space = gym.spaces.Box(low = - np.ones_like(self.raw_min_obs),high = np.ones_like(self.raw_max_obs))
            # action conversion from normalized to raw: y = kx + m
            self.toRawAction = linearMapFcn(-1.0,1.0,self.raw_min_act,self.raw_max_act)
            # action conversion from raw to normalized: nor - normalized
            self.toNormalizedAction = linearMapFcn(self.raw_min_act,self.raw_max_act,-1.0,1.0)            
            # observation from normalized to raw:
            self.toRawObservation = linearMapFcn(-1.0, 1.0,self.raw_min_obs,self.raw_max_obs)            
            # observation from raw to normalized:
            self.toNormalizedObservation = linearMapFcn(self.raw_min_obs,self.raw_max_obs,-1.0,1.0)
        else: # raw action and observation
            self.action_space = gym.spaces.Box(low = self.raw_min_act,high = self.raw_max_act)
            self.observation_space = gym.spaces.Box(low = self.raw_min_obs,high = self.raw_max_obs)
        
    def __del__(self): # Deleting (Calling destructor) 
        print(f'Destructor called, {self.__class__} deleted.')
#         self.close()
    
    @property
    def humanoid_task(self):
        return self._humanoid_task
    
    @humanoid_task.setter
    def humanoid_task(self,is_humanoid:bool):
        print(f"setting humanoid_task={is_humanoid}")
        self._humanoid_task = is_humanoid
        pi = np.pi
        if is_humanoid:
            self.joint_pos_limit = np.array([
                # front left
                [-pi,pi], # 0
                [-pi/2,pi/2], # 1
                [-pi/2,pi/2], # 2
                # front right
                [-pi,pi], # 3
                [-pi/2,pi/2], # 4
                [-pi/2,pi/2], # 5
                # back left
                [-pi/4,pi/4], # 6, reduced range
                [-pi/2-pi/6,-pi/2+pi/4], # 7, reduced range
                [-pi/2,pi/2], # 8
                # back right
                [-pi/4,pi/4], # 9 , reduced range
                [pi/2-pi/4,pi/2+pi/6], # 10, reduced range
                [-pi/2,pi/2], # 11
            ], dtype=np.float32)
        else: # quadruped task
            self.joint_pos_limit = np.array([
                # front left, TODO: CHANGE THIS RANGE
                [-pi,pi], # 0
                [-pi/2,pi/2], # 1
                [-pi/2,pi/2], # 2
                # front right
                [-pi,pi], # 3
                [-pi/2,pi/2], # 4
                [-pi/2,pi/2], # 5
                # back left
                [-pi/2-pi/5, pi/2+pi/5], # 6
                [-pi/2-pi/5, pi/2+pi/5], # 7
                [-pi/2,pi/2], # 8
                # back right
                [-pi/4,pi/4], # 9
                [-pi/2-pi/5, pi/2+pi/5], # 6
                [-pi/2-pi/5, pi/2+pi/5], # 7
            ], dtype=np.float32)
    
    def step(self,action = None):
        if action is not None:
            # map action -> action*max_acton
            cmd_action = np.asarray(action,dtype=np.float32)
            if self.normalize:
                cmd_action = self.toRawAction(cmd_action)
            # # # position difference control
            # cmd_action += self.joint_pos # the actual position
            
            # if len(self.cd.getContactPoints(cmd_action))>0:
            # step_cmd_b = self.packer.pack([self.UDP_MOTOR_POS_COMMEND,time.time(),cmd_action.tolist()])
            step_cmd_b = self.packer.pack([self.UDP_MOTOR_VEL_COMMEND,time.time(),cmd_action.tolist()])
            num_bytes_send = self.server.send(step_cmd_b)
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

        com_acc = np.mean([m[self.ID_com_acc] for m in msg_rec],axis=0)

        com_acc_norm = np.linalg.norm(com_acc)
        r_acc = np.clip(1.3-0.1*com_acc_norm,0,1) # acceleration reward

        # check if out of range
        joint_pos_limit_check = self.joint_pos - np.clip(self.joint_pos,self.joint_pos_limit[:,0],self.joint_pos_limit[:,1])
        joint_out_of_range_norm = np.linalg.norm(joint_pos_limit_check)
        r_joint_limit = max(0,1.0 - joint_out_of_range_norm*10)
        joint_out_of_range = joint_out_of_range_norm>0.1

        # observation = np.stack([np.hstack(msg_i[2:-1]+[msg_i[self.ID_com_pos][2]])
        observation = np.stack([np.hstack(msg_i[2:-1])
                for msg_i in msg_rec]).astype(np.float32) 

        if self.flatten_obs:
            observation = observation.ravel()
        if self.normalize: # normalize the observation
            observation = self.toNormalizedObservation(observation)            

        # x velocity
        com_vel_x = sum([msg_i[self.ID_com_vel][0] for msg_i in msg_rec])/len(msg_rec)
        # r_vel = 0.3*np.clip(com_vel_xy,0,1)+0.7 # velocity reward
        r_vel = 0.5*np.clip(com_vel_x,-0.5,1)+0.6 # velocity reward

    #         print(orientation_z,com_z)
        # r_orientation = max(0,orientation_z)*min(com_z+0.56,1)
        ori = msg_rec_i[self.ID_orientation]

        if self._humanoid_task:
            orientation_z = ori[2] # z_z, local x vector projected to world z direction
            com_z_min = 0.36
            com_z_offset = 0.56
            orientation_z_min = 0.56
        else: # quadruped task
            orientation_z= ori[0]*ori[4] - ori[1]*ori[3] # z_z, local z vector projected to world z direction
            com_z_min = 0.1
            com_z_offset = 0.8
            orientation_z_min = 0.56

        r_orientation = (np.clip(orientation_z*1.02,0,1)**3)*min(com_z+com_z_offset,1)
        # r_orientation = (np.clip(orientation_z*1.02,0,1)**3)#*min(com_z+com_z_offset,1)

        # x = np.linspace(0,1,400)
        # y = np.clip(np.cos(x*np.pi/2)/np.cos(np.pi/180*15),-1,1)**3
        # plt.plot(x,y)
        # print(com_z+0.56)
        # print(msg_rec_i[self.ID_orientation])

        r_quad_ctrl = max(0,1-0.1 * sum(np.square(actuation))) # quad control cost
        reward =  r_orientation*r_quad_ctrl*r_vel*r_joint_limit

    #         reward = orientation_z
        # done = True if ((orientation_z<orientation_z_min)or(com_z<com_z_min)or(self.episode_steps>=self._max_episode_steps)or joint_out_of_range) else False
        done = True if ((orientation_z<orientation_z_min)or(com_z<com_z_min)or(self.episode_steps>=self._max_episode_steps)) else False

        # done = True if (orientation_z<orientation_z_min)or(self.episode_steps>=self._max_episode_steps)or joint_out_of_range else False
        # done = True if (self.episode_steps>=self._max_episode_steps) else False
        t = msg_rec_i[self.ID_t]
        if self.info:
            info = {'t':t,
                    'r_vel':r_vel,
                    'r_orientation':r_orientation,
                    'r_quad_ctrl':r_quad_ctrl,
                    'r_joint_limit':r_joint_limit,
                    'r_acc':r_acc
                    }
        else:
            info = {'t':t}
        # if done:
    #             print(orientation_z,com_z)
        # _ = [m[self.ID['t']] for m in msg_rec]
        # print(_[0]-_[-1])
        return observation,reward,done,info # TODO, change shape
    
    def reset(self):
        self.server.send(self.reset_cmd_b)
        time.sleep(1/50)
        msg_rec = self.receive()
        self.episode_steps = 0
        observation,reward,done,info =  self._processRecMsg(msg_rec)
        self.episode_start_time = info['t']
        return observation
    
    def render(self,mode="human"):
        pass

    def start(self):
        try: # check if the simulation is opened
            msg_rec = self.receive(max_attempts=2,verbose=False)
        except TimeoutError: # program not opened
            # path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"./run_flexipod.bat")
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..\\..\\build\\flexipod.exe")
            # print(path)
            task = subprocess.Popen([path])
            
    def endSimulation(self):
        self.server.send(self.close_cmd_b)

    def receive(self,max_attempts:int = 1000000,verbose=True):
        data = self.server.receive(max_attempts=max_attempts)
        data_unpacked = msgpack.unpackb(data,use_list=False)
        return data_unpacked 
        
    def pause(self):
        """pause the simulation"""
        self.server.send(self.pause_cmd_b)
        
    def resume(self):
        """resume the simulation"""
        self.server.send(self.resume_cmd_b)

def make(**kargs):
    return FlexipodEnv(**kargs)


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