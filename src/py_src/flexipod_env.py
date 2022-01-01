import gym
import subprocess
import time
import numpy as np
from collections import defaultdict
import msgpack
import os
import pickle
import pybullet
from pybullet_utils.bullet_client import BulletClient
from udp_server import UDPServer
from operator import itemgetter

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
    return lambda x:k*x +m

def translation(p):
    """return homogeneous translation matrix"""
    h = np.eye(4)
    h[:3,3] = p
    return h
        
class BulletCollisionDetect:
    """
    Helper class for collision detection using pybullet
    """
    def __init__(s,gui=False,urdf_path="../../data/urdf/12dof/robot.urdf"):
        """
        load urdf and init the collision detection
        """
        try: dir_path = os.path.dirname(os.path.abspath(__file__))
        except NameError: dir_path = os.getcwd()
        urdf_path=os.path.abspath(os.path.join(dir_path,urdf_path))
        # loading using bullet_client
        # ref: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#
        s._p = BulletClient(pybullet.GUI if gui else pybullet.DIRECT)
        s.body_id = s._p.loadURDF(urdf_path,
                              useFixedBase=1,
                              flags=pybullet.URDF_USE_SELF_COLLISION)  # load urdf
        s.dof = s._p.getNumJoints(s.body_id)
        # JOINT_INFO_DICT = {name: i for i, name in enumerate([
        #  "jointIndex", "jointName", "jointType", "qIndex", "uIndex",
        #  "flags", "jointDamping", "jointFriction", "jointLowerLimit",
        #  "jointUpperLimit", "jointMaxForce", "jointMaxVelocity", "linkName",
        #  "jointAxis", "parentFramePos", "parentFrameOrn", "parentIndex"])}
#         joint_info = [s._p.getJointInfo(s.body_id, i) for i in range(s.dof)]
#         s.joint_lower_limit = np.asarray(
#             [j[JOINT_INFO_DICT["jointLowerLimit"]] for j in joint_info])
#         s.joint_upper_limit = np.asarray(
#             [j[JOINT_INFO_DICT["jointUpperLimit"]] for j in joint_info])
#         s.joint_range = s.joint_upper_limit-s.joint_lower_limit

    def getContactPoints(s,joint_pos):
        """get contact points info given joint position in rad
        ref:https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.cb0co8y2vuvc
        """
        for i in range(s.dof):
            s._p.resetJointState(s.body_id, i, joint_pos[i],0)
        s._p.performCollisionDetection()
        return s._p.getContactPoints()
        
    def __del__(s):
        s._p.disconnect()
#         print("deleted")


class FlexipodEnv(gym.Env):
    """
    openai gym compatible environment for the simulated 12DOF flexipod
    ref: https://github.com/openai/gym/blob/master/gym/core.py
    """

    def __init__(
        s, # self
        dof:int = 12, # degress of freedom of the robot
        num_observation:int=5, # number of observation in series
        num_sensors:int = 0, # num of spring strain sensors
        normalize:bool = True, # whether to normlaize the observation and action
        max_joint_vel:float = 10, # maximum joint velocity rad/s
        humanoid_task:bool = True, # if true, humanoid, else qurdurped
        max_episode_steps:int = 2000,
        max_action:float = 10, # [rad/s], velocity control
        flatten_obs:bool = True, # whether to flatten observation
        info = False, # bool flag to send extra information when processing messages
        local_address = ("127.0.0.1",33300),
        remote_address = ("127.0.0.1",33301),
        robot_folder:str = "../../robot/v11" # root folder for the robot
        ):
        super().__init__() # init gym env
        
        # collision detection with pybullet
        s.cd = BulletCollisionDetect(urdf_path=robot_folder+"/urdf/robot.urdf") 
        
        # general variables
        s.dof = dof #    num joints
        s.num_observation = num_observation
        s.normalize = normalize
        s.flatten_obs = flatten_obs # True
        s.info = info # bool flag to send info
        # https://github.com/openai/gym/blob/master/gym/wrappers/time_limit.py
        s._max_episode_steps = max_episode_steps # maximum episode steps
        # general variables (mutable)
        s.joint_pos = np.empty(dof,dtype=np.float32) # joint position [rad]
        s.episode_step = 0 # curret step in an episode
        
        s.local_address = local_address
        s.remote_address = remote_address
        
        # name of the returned message from simulation
        s.REC = np.array([
            # name,         size,         min,          max
            ("header",       1,         -32768,         32768        ),
            ("t",            1,         0,              np.inf       ), # simulation time [s]
            ("joint_pos",    dof*2,     -1.,            1.           ), # joint cos(angle) sin(angle) [rad]
            ("joint_vel",    dof,       -max_joint_vel, max_joint_vel), # joint velocity [rad/s]
            ("joint_torque", dof,       -1.,            1.           ), # joint torque [-1,1]
            # ("joint_cmd",    dof,       -np.pi,         np.pi        ), # joint command from policy
            ("orientation",  6,         -1.,            1.           ), # base link (body) orientation
            ("ang_vel",      3,         -30.,           30.          ), # base link (body) angular velocity [rad/s]
            ("com_acc",      3,         -30.,           30.          ), # base link (body) acceleration
            ("com_vel",      3,         -2.,            2.           ), # base link (body) velocity
            ("com_pos",      3.,        -1.,            1.           ), # base link (body) position
            ("constraint", dof+1,         0,            1            ), # selected constraint_force/total_weight
            ("spring_strain",num_sensors,-0.005,       0.005         ), # selected spring strain
        ],dtype=[('name', 'U20'), ('size', 'i4'), ('min', 'f4'),('max', 'f4')])
        
        s.ID =defaultdict(None,{name: k  for k,name in enumerate(s.REC["name"])})
        s.ID_t = s.ID['t']
        s.ID_joint_pos = s.ID['joint_pos']
        s.ID_joint_vel = s.ID['joint_vel']
        s.ID_joint_torque = s.ID['joint_torque']
        s.ID_orientation = s.ID['orientation']
        s.ID_com_vel = s.ID['com_vel']
        s.ID_com_acc = s.ID['com_acc']
        s.ID_com_pos = s.ID['com_pos']
        s.ID_ang_vel = s.ID['ang_vel']
        s.ID_constraint = s.ID['constraint']
        
        s.SENSOR_DICT = {
        # part_id: pos_offset (sensered position offset measured from the sensor coordinate)
            9: (0,0,-100),  # left_foot
            12: (0,0,-100)  # right_foot
        }
        for key in s.SENSOR_DICT: # convert to numpy array
            s.SENSOR_DICT[key] = np.array(s.SENSOR_DICT[key],dtype=np.float32)
        s.sensor_id = tuple(s.SENSOR_DICT.keys())# left_foot, right_foot, index of the mass part id
        s.sensor_pos = tuple(s.SENSOR_DICT.values()) # sensered position offset measured from the sensor coordinate
        s.sensor_distance = tuple(np.linalg.norm(pos) for pos in s.sensor_pos)  # distance to joint
        s.t_sensor = tuple(translation(p) for p in s.sensor_pos) # sensor transform
        
        # selected observation
        OBS_DICT = { # dictionary is ordered since python 3.6
            "joint_pos":None,                    #0
            "joint_vel":None,                    #1
            "joint_torque":None,                 #2
            # "joint_cmd":None,                    #
            "orientation":None,                  #3
            "ang_vel":None,                      #4
            "com_acc":None,                      #5
            # "com_vel":None,                    #6
            # "com_pos":(2,), # com-z            #7
            "constraint":s.sensor_id,            #8
            "spring_strain":None,                #9
        }
        
        s.humanoid_task = humanoid_task
            
        #------------task vector setup ---------------------------#
        # vel reward params
        # s.desired_vel = 0.2 # desired com vel [m/s]
        s.min_desired_vel = -0.25 # desired com vel [m/s] lower bound
        s.max_desired_vel = 0.25 # desired com vel [m/s] upper bound
        # conversion raw<->normalized desired_vel
        s.toRawDesiredVel = linearMapFcn(0,1,s.min_desired_vel,s.max_desired_vel) # to raw
        s.toNorDesiredVel =  linearMapFcn(s.min_desired_vel,s.max_desired_vel,0,1) # to normalized
        
        # frequency
        # s.gait_frequency = 1.5
        s.min_gait_frequency = 1.25 # Hz
        s.max_gait_frequency = 1.75 # Hz
        # conversion raw<->normalized gait_frequency
        s.toRawGaitFrequency = linearMapFcn(0,1,s.min_gait_frequency,s.max_gait_frequency) # to raw
        s.toNorGaitFrequency = linearMapFcn(s.min_gait_frequency,s.max_gait_frequency,0,1) # to normalized
        
        # normalized phase
        s.min_phase = 0
        s.max_phase = 1
        #----------------------------------------------------------#
        # TODO-> position control
        max_action = 0.2*max_action
        
        s._initObsACt(max_action,s.REC,OBS_DICT) # initialize the observation and action space
        
        s._implInitServer()

        s._initRobotGraph(robot_folder)
        # start the simulation
        s.start()
        
    def _implInitServer(s):
        """implementation specific initiailzation of the server"""
        # defualt packer # https://msgpack-python.readthedocs.io/en/latest/api.html#msgpack.Packer
        s.packer = msgpack.Packer(use_single_float=True, use_bin_type=True)
        # name of command message
        s.CMD_NAME = ("header", "t", "cmd")
        s.CMD_ID = defaultdict(None,{name: k  for k,name in enumerate(s.CMD_NAME)})
        # UDP header
        s.UDP_TERMINATE = int(-1)
        s.UDP_PAUSE = int(17)
        s.UDP_RESUME = int(16)
        s.UDP_RESET = int(15)
        s.UDP_ROBOT_STATE_REPORT = int(14)
        s.UDP_MOTOR_VEL_COMMEND = int(13)
        s.UDP_STEP_MOTOR_VEL_COMMEND = int(12)
        s.UDP_MOTOR_POS_COMMEND = int(11)
        s.UDP_STEP_MOTOR_POS_COMMEND = int(10)
        # simulation specific commend
        s.reset_cmd_b =  s.packer.pack([s.UDP_RESET])
        s.pause_cmd_b =  s.packer.pack([s.UDP_PAUSE])
        s.resume_cmd_b = s.packer.pack([s.UDP_RESUME])
        s.close_cmd_b =  s.packer.pack([s.UDP_TERMINATE])
        # udp server
        s.server = UDPServer(s.local_address,s.remote_address)
        
        
    def _initRobotGraph(s,robot_folder):
        """initialized and load the robot graph from file"""
        if not os.path.isabs(robot_folder): # absolute path or not
            try:base_dir =  os.path.dirname(__file__) # file location
            except NameError:base_dir = os.getcwd()
            robot_folder = os.path.abspath(os.path.join(base_dir,robot_folder))
        
        _t= time.time()
        with open(f"{robot_folder}//robot.pickle","rb") as f:
            s.graph = pickle.load(f)
        print(f"Loading robot graph takes {time.time()-_t:.1f} s.")
        
        s.node_sensor = itemgetter(*s.sensor_id)(s.graph.nodes)#node_left_foot,node_right_foot
        
        # t= eye(4), world thransform equivalent to root space transfrom
        s.joint_pos = s.graph.getJointPosArrFast()
        s.graph.setJointPosArrFast(s.joint_pos) # set initial joint pos to graph.joint_pos
        s.graph.updateWorldTransformFast(np.eye(4))
        
        # foot sensor position in the robot root space
        lf_sensor_pos = (s.node_sensor[0]["world_transform"]@s.t_sensor[0])[0:3,3] # left foot
        rf_sensor_pos = (s.node_sensor[1]["world_transform"]@s.t_sensor[1])[0:3,3] # right foot
        with np.printoptions(precision=2, suppress=True, threshold=5):
            print(f"robot feet sensor pos: left:{lf_sensor_pos} right:{rf_sensor_pos}")
        
        from phase_indication import phaseIndicatorPair
        s.phase_indicator_p0 = phaseIndicatorPair(a=0.3, b=0.7, s=0.05, t0=0.25, t1=0.75, ys = -1., y0=0.)
        s.phase_indicator_p1 = phaseIndicatorPair(a=0.25, b=0.75, s=0.05, t0=0.25, t1=0.75, ys = 2., y0=-1.)
    
    def cyclicReward(s,t_normalized):
        s.graph.setJointPosArrFast(s.joint_pos)
        s.graph.updateWorldTransformFast(np.eye(4))

        # indicator for foot force
        i0_0,i0_1= s.phase_indicator_p0.get(t_normalized)
        # indicator for foot forward displacemnt relative to body, input to the example gait generator
        i1_0,i1_1= s.phase_indicator_p1.get(t_normalized) 

        # s.sensor_force
        # when left foot is raised, left foot sensor force should be 0, 
        # penalize foot force cyclically 
        c0_0 = min(1,s.sensor_force[0]) # left foot, keep max at 1
        c0_1 = min(1,s.sensor_force[1]) # right foot
        r0 = i0_0*c0_0+i0_1*c0_1
        # r0_0,r_01 = c0_0*i0_0,c0_1*i0_1
            
        # body space left  foot sensor forward displacement / normalizaton_coefficient
        c1_0 = -(s.node_sensor[0]["world_transform"]@s.t_sensor[0])[2,3]/s.sensor_distance[0]
        # body space right foot sensor forward displacement / normalizaton_coefficient
        c1_1 = -(s.node_sensor[1]["world_transform"]@s.t_sensor[1])[2,3]/s.sensor_distance[1]
        # print(f"c1_0, c1_1={c1_0:.3f},{c1_1:.3f}")
        c1_0 = max(-1,min(1,c1_0)) # clamp to [-1,1]
        c1_1 = max(-1,min(1,c1_1)) # clamp to [-1,1]
        r1 = i1_0*c1_0+i1_1*c1_1
        # r1_0,r1_1 = i1_0*c1_0,i1_1*c1_1
        
        # # indicator 0
        # print(f"i0_0, i0_1={i0_0:.3f}, {i0_1:.3f}")
        # print(f"c0_0, c0_1={c0_0:.3f}, {c0_1:.3f}")
        # print(f"r0_0, r0_1={r0_0:.3f}, {r0_1:.3f}")
        # print(f"r0={r0:+5.3f}")
        # # indicator 1
        # print(f"i1_0, i1_1={i1_0:+5.3f}, {i1_1:+5.3f}")
        # print(f"c1_0, c1_1={c1_0:+5.3f},{c1_1:+5.3f} (clamp to [-1,1])")
        # print(f"r1_0, r1_1={r1_0:+5.3f}, {r1_1:+5.3f}")
        # print(f"r1={r1:+5.3f}")
        
        r_cyclic = (0.8+r0)*(1+r1)
        return r_cyclic
        
    def _initObsACt(s,max_action,REC,OBS_DICT):
        """helper function to initialize the observation and action space"""
        for key in list(OBS_DICT): # remove item with zero size
            if s.REC["size"][s.ID[key]]==0:
                print(f"{key} size==0, removed from OBS_DICT")
                OBS_DICT.pop(key)
        s.OBS_NAME = tuple(OBS_DICT.keys()) # keys in OBS_DICT
        s.OBS_IDS = tuple(s.ID[n] for n in s.OBS_NAME) # indices of the OBS
        s.OBS_ITEM_SLICE = tuple(OBS_DICT.values()) # indices of each item in OBS_DICT
        
        def conditional_flatten(msg_i):
            """helper to conditionally flatten a list of list"""
            flattend = []
            for item,item_slice in zip(itemgetter(*s.OBS_IDS)(msg_i),s.OBS_ITEM_SLICE):
                try: # check if item is not single number
                    len(item) # len>1
                    if item_slice is not None:
                        item = itemgetter(*item_slice)(item)
                    try:
                        len(item) # len>1
                        flattend.extend(item)# not single number
                    except TypeError: # single number
                        flattend.append(item) # single number
                except TypeError: # single number
                    flattend.append(item)
            return flattend
        s.conditional_flatten = conditional_flatten
        
        # raw min/max observation
        s.raw_min_obs = np.hstack([ # raw max observation (from simulation)
            np.ones(REC["size"][s.ID[n]] if OBS_DICT[n] is None else len(OBS_DICT[n])
                   )*REC["min"][s.ID[n]]for n in s.OBS_NAME]).astype(np.float32)
        s.raw_max_obs = np.hstack([ # raw max observation (from simulation)
            np.ones(REC["size"][s.ID[n]] if OBS_DICT[n] is None else len(OBS_DICT[n])
                   )*REC["max"][s.ID[n]]for n in s.OBS_NAME]).astype(np.float32)

        # multiple observation per network input
        s.raw_min_obs = np.tile(s.raw_min_obs,(s.num_observation,1))
        s.raw_max_obs = np.tile(s.raw_max_obs,(s.num_observation,1))
        
        
        if s.flatten_obs:# whether to flatten the observation
            s.raw_min_obs = s.raw_min_obs.ravel()
            s.raw_max_obs = s.raw_max_obs.ravel()
            
            # TODO MAKE IT CYCLIC
            #                                      phase, com_desired_vel, gait_frequency
            s.raw_min_obs = np.append(s.raw_min_obs,[0,s.min_desired_vel,s.min_gait_frequency]).astype(np.float32)
            s.raw_max_obs = np.append(s.raw_max_obs,[1,s.max_desired_vel,s.max_gait_frequency]).astype(np.float32)
            
        # raw min/max action for all motors # [rad/s]
        s.raw_min_act = - max_action*np.ones(s.dof,dtype=np.float32)
        s.raw_max_act =   max_action*np.ones(s.dof,dtype=np.float32)
        
        if s.normalize: # conditionally normalize the action space
            s.action_space = gym.spaces.Box(low = - np.ones_like(s.raw_min_act),high = np.ones_like(s.raw_max_act))
            s.observation_space = gym.spaces.Box(low = - np.ones_like(s.raw_min_obs),high = np.ones_like(s.raw_max_obs))
            # action conversion from normalized to raw: y = kx + m
            s.toRawAction = linearMapFcn(-1.0,1.0,s.raw_min_act,s.raw_max_act)
            # action conversion from raw to normalized: nor - normalized
            s.toNormalizedAction = linearMapFcn(s.raw_min_act,s.raw_max_act,-1.0,1.0)            
            # observation from normalized to raw:
            s.toRawObservation = linearMapFcn(-1.0, 1.0,s.raw_min_obs,s.raw_max_obs)            
            # observation from raw to normalized:
            s.toNormalizedObservation = linearMapFcn(s.raw_min_obs,s.raw_max_obs,-1.0,1.0)
        else: # raw action and observation
            s.action_space = gym.spaces.Box(low = s.raw_min_act,high = s.raw_max_act)
            s.observation_space = gym.spaces.Box(low = s.raw_min_obs,high = s.raw_max_obs)
        
    def __del__(s): # Deleting (Calling destructor) 
        print(f'Destructor called, {s.__class__} deleted.')

    def exampleAction(s):
        pi = np.pi
        p_f_b0 = pi/24 # font arm body-0
        p_f_01 = pi/3 # front arm 01
        p_f_12 = -pi/12 # front arm 12
        p_b0_base = pi/8 # back leg body-0
        
        # indicator for foot force
        i0_0,i0_1= s.phase_indicator_p0.get(s.t_normalized)
        # indicator for foot forward displacemnt relative to body, input to the example gait generator
        i1_0,i1_1= s.phase_indicator_p1.get(s.t_normalized)
        
        i0 = i1_0 # use phase_indicator_pair_1
        i1 = i1_1
        
        p_bl_b0 =p_b0_base+i0*0.08
        p_br_b0 =p_b0_base+i1*0.08

        p_bl_01 = -p_bl_b0*2
        p_br_01 = -p_br_b0*2

        p_bl_12 = p_bl_b0
        p_br_12 = p_br_b0

        joint_pos = np.array([
            # front left
                -p_f_b0,
                -p_f_01,
                -p_f_12,
            # front right
                p_f_b0,
                p_f_01,
                p_f_12,
            # back left
                -p_bl_b0 - i0*0.08 ,
                -p_bl_01,
                -p_bl_12,
            #back right
                p_br_b0 + i1*0.08,
                p_br_01,
                p_br_12,
        ])
        
        delta_p = 1*(joint_pos-s.joint_pos)
        return delta_p

    @property
    def humanoid_task(s):
        return s._humanoid_task
    
    @humanoid_task.setter
    def humanoid_task(s,is_humanoid:bool):
        print(f"setting humanoid_task={is_humanoid}")
        s._humanoid_task = is_humanoid
        pi = np.pi
        if is_humanoid:
            # s.joint_pos_limit = np.array([ v9
            #     # front left
            #     [-pi/1.25,pi/1.25], # 0
            #     [-pi/2,pi/2], # 1
            #     [-pi/2,pi/2], # 2
            #     # front right
            #     [-pi/1.25,pi/1.25], # 3
            #     [-pi/2,pi/2], # 4
            #     [-pi/2,pi/2], # 5
            #     # back left
            #     [-pi/3,pi/3], # 6, reduced range
            #     [-pi/2-pi/6,-pi/2+pi/4], # 7, reduced range
            #     [-pi/2,pi/12], # 8
            #     # back right
            #     [-pi/3,pi/3], # 9 , reduced range
            #     [pi/2-pi/4,pi/2+pi/6], # 10, reduced range
            #     [-pi/12,pi/2], # 11
            # ], dtype=np.float32)
            # s.com_z_min = 0.34
            # s.com_z_offset = 0.62
            # s.orientation_z_min = 0.56
            s.joint_pos_limit = np.array([ # v11
                # front left
                [-pi/1.25,pi/1.25], # 0
                [-pi/2,pi/2], # 1
                [-pi/2,pi/2], # 2
                # front right
                [-pi/1.25,pi/1.25], # 3
                [-pi/2,pi/2], # 4
                [-pi/2,pi/2], # 5
                # back left
                [-pi/3,pi/3], # 6, reduced range
                [0,    pi/2], # 7, reduced range
                [-pi/3,pi/3], # 8
                # back right
                [-pi/3,pi/3], # 9 , reduced range
                [-pi/2,   0], # 10, reduced rrange
                [-pi/3,pi/3], # 11
            ], dtype=np.float32)
            s.com_z_min = 0.34
            s.com_z_offset = 0.57
            s.orientation_z_min = 0.56
            
        else: # quadruped task
            s.joint_pos_limit = np.array([
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
            s.com_z_min = 0.1
            s.com_z_offset = 0.8
            s.orientation_z_min = 0.56
            
        s.joint_pos_min = s.joint_pos_limit[:,0]
        s.joint_pos_max = s.joint_pos_limit[:,1]
    
    def step(s,action = None):
        """ act and recive observation from the env"""
        s.act(action) # act
        return s.observe()  # observe
    
    def act(s,action=None):
        """do action"""
        if action is not None:
            # map action -> action*max_acton
            cmd_action = np.asarray(action,dtype=np.float32)
            if s.normalize:
                cmd_action = s.toRawAction(cmd_action)
            # # position difference control
            cmd_action += s.joint_pos # the actual position
            # clamp to min and max joint position
            cmd_action = np.clip(cmd_action,s.joint_pos_min,s.joint_pos_max)
            # if len(s.cd.getContactPoints(cmd_action))>0:
            s._impl_SendAction(cmd_action.tolist())
            
    def observe(s):
        """
        observe outcome from the env,
        return observation,reward,done,info
        """
        msg_rec = s._impl_Receive()
        s.episode_step+=1 # update episodic step counter
        observation = s._impl_ProcessObservation(msg_rec)
        reward,done,info = s.getReward()
        return observation,reward,done,info
    
    def _impl_SendAction(s,cmd_action:list):
        """
        implementation specific function to serialize and send the action 
        processed by act() to the env.
        """
        step_cmd_b = s.packer.pack([s.UDP_MOTOR_POS_COMMEND,time.time(),cmd_action])
        # step_cmd_b = s.packer.pack([s.UDP_MOTOR_VEL_COMMEND,time.time(),cmd_action])
        num_bytes_send = s.server.send(step_cmd_b)
    
    def _impl_Receive(s,timeout:float = 4):
        """Implementation specific receive messages from the env"""
        data = s.server.receive(timeout=timeout)
        data_unpacked = msgpack.unpackb(data,use_list=False)
        return data_unpacked 
    
    def _impl_ProcessObservation(s,msg):
        """
        implemtation specific function to process the received messages
        from the environment. variables updated and stored in s are:
            s.t : time [s]
            s.t_normalized: time normalized by gait frequency, [unitless]
            s.episode_start_time: episode start time [s]
            s.com_pos: 1x3 com position measured at world space [m]
            s.com_vel: 1x3 com velocity measured at world space
            s.com_acc: 1x3 com acceleration measured at world space [m/s^2]
            s.joint_pos: 1xdof joint position [rad]
            s.joint_torque: 1xdof joint torque [Nm]
            s.ang_vel: angular velocity [rad/s] measured at body space
            s.oreint: 6D orientation vector masured at world space 
                      (first two columns of the rotation matrix)
            s.sensor_force: measured normalized constraint force
            s.desired_vel_now: current desired velocity [m/s]
        returns: 
            observation: observation plus the task vector
        """
        msg_i = msg[0]
        
        # time
        s.t = msg_i[s.ID_t]
        # update episode_start_time
        if s.episode_step ==0: 
            s.episode_start_time = s.t
        #  time normalized by gait frequency, [unitless]
        s.t_normalized = ((s.t-s.episode_start_time)*s.gait_frequency)%1 
        
        # 1x3 com position measured at world space [m]
        s.com_pos = msg_i[s.ID_com_pos]
        # 1x3 com velocity measured at world space [m/s]
        s.com_vel = np.mean([m[s.ID_com_vel] for m in msg],axis=0)
        # 1x3 com acceleration measured at world space [m/s^2]
        s.com_acc = np.mean([m[s.ID_com_acc] for m in msg],axis=0)
        
        # joint position (sin,cos->rad)
        joint_pos = msg_i[s.ID_joint_pos]
        _ = np.arctan2(joint_pos[1::2],joint_pos[::2],s.joint_pos,dtype=np.float32) # setting s.joint_pos
        # print(f"s.joint_pos ={s.joint_pos}")
        # measured joint torque
        s.joint_torque = msg_i[s.ID_joint_torque] # joint torque (size=dof) of the latest observation
        
        # angular velocity [rad/s] measured at body space
        s.ang_vel = np.mean([m[s.ID_ang_vel] for m in msg],axis=0)
        # 6D orientation vector masured at world space (first two columns of the rotation matrix)
        s.orient = msg_i[s.ID_orientation]

        # measured normalized constraint force
        s.sensor_force = itemgetter(*s.sensor_id)(msg_i[s.ID_constraint])
        
        observation = np.stack([s.conditional_flatten(msg_i) for msg_i in msg]).astype(np.float32)
        
        # ramping up desired speed
        s.desired_vel_now = min(s.episode_step/160.,1)*s.desired_vel
        
        if s.flatten_obs:
            observation = observation.ravel()
            observation = np.append(observation,[s.t_normalized,s.desired_vel_now,s.gait_frequency]).astype(np.float32)
            
        if s.normalize: # normalize the observation
            observation = s.toNormalizedObservation(observation)
        
        return observation  
            
    def getReward(s):
        """compute the reward at current step, 
        assuming _impl_ProcessObservation() is called beforehand"""
        # acceleration cost
        com_acc_norm = np.linalg.norm(s.com_acc)
        r_acc = max(0,min(1.3-0.1*com_acc_norm,1)) 

        # angular velocity cost
        ang_vel_norm =  np.linalg.norm(s.ang_vel)
        r_ang_vel = max(0,min(1-0.1*ang_vel_norm,1))
        
        # check if out of range
        joint_pos_limit_check = s.joint_pos - np.clip(s.joint_pos,s.joint_pos_limit[:,0],s.joint_pos_limit[:,1])
        joint_out_of_range_norm = np.linalg.norm(joint_pos_limit_check)
        r_joint_limit = max(0,1.0 - joint_out_of_range_norm*10)
        joint_out_of_range = joint_out_of_range_norm>0.1

        # velocity cost
        com_vel_x = s.com_vel[0] # x velocity
        # r_vel = 5*np.clip(com_vel_x,-0.2,0.5)+1.0 # velocity reward worked
        r_vel = 2.0-min(0.4,abs(s.desired_vel_now-com_vel_x))*5 # com velocity reward
    
        if s._humanoid_task:
            orientation_z = s.orient[2] # z_z, local x vector projected to world z direction
        else: # quadruped task
            orientation_z= s.orient[0]*s.orient[4] - s.orient[1]*s.orient[3] # z_z, local z vector projected to world z direction

        com_z = s.com_pos[2] # com z pos
        r_orientation = (max(0,min(orientation_z*1.02,1))**3)*min(com_z+s.com_z_offset,1)
        # print(orientation_z,com_z)
        
        # quad control cost
        r_quad_ctrl = max(0,1-0.05 * sum(np.square(s.joint_torque))) 
        
        #for computing cyclic reward
        r_cyclic =  s.cyclicReward(s.t_normalized)
        
        reward =  r_orientation*r_quad_ctrl*r_vel*r_ang_vel*r_joint_limit*r_cyclic
        
        done = True if ((orientation_z<s.orientation_z_min)or(com_z<s.com_z_min)) else False
        # done = True if ((orientation_z<s.orientation_z_min)or(com_z<s.com_z_min)or(s.episode_steps>=s._max_episode_steps)) else False
        # done = True if ((orientation_z<s.orientation_z_min)or(com_z<s.com_z_min)or(s.episode_steps>=s._max_episode_steps) or joint_out_of_range) else False

        if s.info:
            info = {'t':s.t,
                    'com':s.com_pos,
                    'com_acc':s.com_acc,
                    'tn':s.t_normalized,
                    'r_cyclic':r_cyclic,
                    'r_vel':r_vel,
                    'r_ang_vel':r_ang_vel,
                    'r_orientation':r_orientation,
                    'r_quad_ctrl':r_quad_ctrl,
                    'r_joint_limit':r_joint_limit,
                    'r_acc':r_acc,
                    }
        else:
            info = {'t':s.t}
        return reward,done,info        
        
    
    def reset(s, desired_vel=None,gait_frequency=None):
        """reset the env"""
        # set desired_vel
        if desired_vel is None:
            desired_vel = np.random.random() # normalized
            # desired_vel = np.random.uniform(0.5,1.0)
        s.desired_vel = s.toRawDesiredVel(desired_vel)

        if gait_frequency is None:
            gait_frequency = np.random.random()
        s.gait_frequency = s.toRawGaitFrequency(gait_frequency)
        
        s.server.send(s.reset_cmd_b)
        time.sleep(1/20)
        msg_rec = s._impl_Receive()
        s.episode_step = 0 # curret step in an episode
        observation = s._impl_ProcessObservation(msg_rec)
        return observation
    
    def render(s,mode="human"):
        pass

    def start(s):
        try: # check if the simulation is opened
            msg_rec = s._impl_Receive(timeout=0.5)
        except Exception: # program not opened
            try: dir_path = os.path.dirname(os.path.abspath(__file__))
            except NameError: dir_path = os.getcwd()
            # path = os.path.join(dir_path,"./run_flexipod.bat")
            path = os.path.join(dir_path,"..\\..\\build\\flexipod.exe")
            # print(path)
            task = subprocess.Popen([path])
            
        
    def pause(s):
        """pause the env"""
        s.server.send(s.pause_cmd_b)
        
    def resume(s):
        """resume the env"""
        s.server.send(s.resume_cmd_b)
        
    def terminate(s):
        """terminate the env"""
        s.server.send(s.close_cmd_b)


class FlexipodHumanoidV10(FlexipodEnv):
    @property
    def humanoid_task(s):
        return s._humanoid_task
    
    @humanoid_task.setter
    def humanoid_task(s,is_humanoid:bool):
        print(f"setting humanoid_task={is_humanoid}")
        s._humanoid_task = is_humanoid
        pi = np.pi
        if is_humanoid:
            s.joint_pos_limit = np.array([
                # front left
                [-pi,pi],     # 0
                [-pi/2,pi/2], # 1
                # front right
                [-pi,pi],     # 2
                [-pi/2,pi/2], # 3
                # back left
                [-pi/3,pi/3], # 4, reduced range
                [-pi/2-pi/6,-pi/2+pi/4], # 5, reduced range
                [-pi/3,pi/3], # 6
                [-pi/3,pi/3], # 7
                # back right
                [-pi/3,pi/3], # 8 , reduced range
                [pi/2-pi/4,pi/2+pi/6], # 9, reduced range
                [-pi/3,pi/3], # 10
                [-pi/3,pi/3], # 11
            ], dtype=np.float32)
            
            s.com_z_min = 0.42
            s.com_z_offset = 0.52
            s.orientation_z_min = 0.6
        else:
            raise NotImplementedError
        s.joint_pos_min = s.joint_pos_limit[:,0]
        s.joint_pos_max = s.joint_pos_limit[:,1]
        print(f"init FlexipodHumanoid,com_z_min = {s.com_z_min}")

    # def __init__(s,**kwargs):
    #      super().__init__(**kwargs)
    #      s.humanoid_task = True


def make(**kargs):
    return FlexipodEnv(**kargs)
    # return FlexipodHumanoid(**kargs)


if __name__ == '__main__':
    # env = FlexipodEnv(dof = 12)
    env = make()
    env.info=True
    env.reset()
    observation,reward,done,info = env.step()
    print(info)
    
#     while True:
#         action = env.action_space.sample()
#         env.step(action)

        # time.sleep(0.0001)
    # print(subprocess.Popen(["cmd"], shell=True))
    # print(subprocess.Popen(["set CUDA_VISIBLE_DEVICES=0"], shell=True))

    # env.startSimulation()
    # print("exit python")