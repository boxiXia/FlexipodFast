import numpy as np
import pybullet as p
import time
import pybullet_data
from scipy.spatial.transform import Rotation

##########################################
urdf_path = "../../data/urdf/12dof/robot.urdf"
####################################################
use_fixed_base=0

gui = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation(physicsClientId=gui)
p.setGravity(0, 0, -10)
# robot_id = p.loadURDF(urdf_path, [0, 0, 0.5], physicsClientId=gui,useFixedBase=1,flags = p.URDF_USE_SELF_COLLISION)
robot_id = p.loadURDF(urdf_path, [0, 0, 0.5], physicsClientId=gui,useFixedBase=use_fixed_base,flags = p.URDF_USE_SELF_COLLISION)

plane = p.loadURDF("plane.urdf")


joint_pos = np.array([
    # front left
    -np.pi/2,               # 0
    -np.pi/2,        # 1
    0,               # 2
    # front right
    np.pi/2,               # 3 
    np.pi/2,         # 4  
    0,               # 5
    # back left
    -np.pi/2,               # 6     
    -np.pi/2,        # 7   
    0,               # 8
    # back right
    np.pi/2,               # 9        
    np.pi/2,         # 10 
    0                # 11   
]) 
dof = len(joint_pos)
for k in range(dof):
    p.resetJointState(robot_id, k, targetValue=joint_pos[k])
    
# quat = Rotation.from_euler('xyz',[0,-np.pi/2,0]).as_quat()
# p.resetBasePositionAndOrientation(robot_id, [0, 0, 0.5], quat)

p.setJointMotorControlArray(robot_id, list(range(dof)), controlMode= p.POSITION_CONTROL,
                                    targetPositions=joint_pos)
# for k in range(1000):
#     p.stepSimulation(physicsClientId=gui)
    
# # ##################################################
while (p.getConnectionInfo(physicsClientId=gui)["isConnected"]):
    #p.setJointMotorControlArray(robot_id,np.arange(p.getNumJoints(robot_id)),p.POSITION_CONTROL,joint_pos)
    p.stepSimulation(physicsClientId=gui)
    #time.sleep(0.01)

p.disconnect()