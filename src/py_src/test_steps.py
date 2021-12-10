from flexipod_env import FlexipodEnv
import numpy as np
env = FlexipodEnv()
env.info=True
obs = env.reset()


pi = np.pi


def exampleAction(self,
                  n0=pi/60,
                  n1=pi/9.5,
                  n2=pi/12
                  ):
    

    p_f_b0 = -pi/6  # font arm body-0
    p_f_01 = pi/3  # front arm 01
    p_f_12 = -pi/4  # front arm 12

#     n0 = pi/60  # body incline
#     # n0 = 0 # body incline

#     n1 = pi/9.5
#     n2 = pi/12
    while True:
        for p_bl_b0, p_br_b0 in zip([n2, n1], [n1, n2]):

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
                -p_bl_b0-n0,
                -p_bl_01,
                -p_bl_12,
                # back right
                p_br_b0+n0,
                p_br_01,
                p_br_12
            ])

            while True:
                delta_p = 1.5*(joint_pos-self.joint_pos)
                if np.linalg.norm(delta_p) > 4e-2:
                    yield delta_p
                else:
                    yield delta_p
                    break


########################################################################        
max_steps = 1000

def collect_sample(n):
    obs = env.reset()
    exp = exampleAction(env,*n)
    r = 0
    for k in range(max_steps):
        action = next(exp)
        observation,reward,done,info = env.step(action)
        r+=reward
#         print(env.episode_steps)
        if done:
            return r
    return r

u = np.array([pi/60,pi/9.5,pi/12])
s = np.array([pi/30,pi/6,pi/6])


num_population = 20
r_list = np.zeros(num_population)
rng = np.random.default_rng()
for k_generation in range(10):
    vals = rng.standard_normal((num_population,len(u)))
    ns = u + vals*s # sample
    for k in range(num_population):
        r = collect_sample(ns[k])
        r_list[k] = r
        print(k,r)