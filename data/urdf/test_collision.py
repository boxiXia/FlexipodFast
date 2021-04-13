import time
import numpy as np
import os
import pybullet as p
import pybullet_data
import random

PI = np.pi

robot_path = os.path.join(os.path.dirname(__file__), "test/robot.urdf")
physicsClient = p.connect(p.GUI)
mode = p.POSITION_CONTROL
p.setGravity(0, 0, 0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.loadURDF("plane.urdf")
robot_start_pos = [0, 0, 0.6]
robot_start_orientation = p.getQuaternionFromEuler([0, -PI / 2, 0])  # make robot vertical
robot_id = p.loadURDF(robot_path, robot_start_pos, robot_start_orientation, useFixedBase=1)  # load urdf
joint_pos = [0 for _ in range(12)]
control_index = [i for i in range(12)]
initial_state = p.saveState()

# motor index
# 0 1 2 upper left
# 3 4 5 upper right
# 6 7 8 lower left
# 9 10 11 lower right
# joint limit
# 0: [-PI, PI] 3: [-PI, PI] 6: [-PI/2, PI] 9: [-PI, PI/2]
# 1: [-7/12 * PI, 7/12 * PI] 4: [-7/12 * PI, 7/12 * PI]  7: [-7/12 * PI, 7/12 * PI] 10: [-7/12 * PI, 7/12 * PI]
# 2: [4/9 * PI, -4/9 * PI] 5: [4/9 * PI, -4/9 * PI] 8: [4/9 * PI, -4/9 * PI] 11: [4/9 * PI, -4/9 * PI]

leg_index = [[0, 1, 2],
             [3, 4, 5],
             [6, 7, 8],
             [9, 10, 11]]
while True:

    for i in range(12):
        p.resetJointState(robot_id, control_index[i], joint_pos[i])
    p.stepSimulation()
    test = p.getClosestPoints(robot_id, robot_id, 1, linkIndexA=11, linkIndexB=-1)
    print(test[0][8])
