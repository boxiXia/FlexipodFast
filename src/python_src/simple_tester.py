import sys
import os

from omegaconf import DictConfig

import hydra
from hydra.utils import instantiate

# import dmc2gym

# PACKAGE_PARENT = 
# PACKAGE_PARENT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..\\"))
# sys.path.append(PACKAGE_PARENT)

# from flexipod_env import FlexipodEnv

# import utils
# from python.flexipod_env import FlexipodEnv
# import python.pytorch_sac.video


# class makeEnv(object):
#     def __init__(self,domain_name):
#         self.domain_name=domain_name

def make(**kargs):
    print(kargs)
    return 0


# @hydra.main(config_path='config/train.yaml', strict=True)
# @hydra.main(config_path="pytorch_sac//config",config_name='train')
@hydra.main(config_path=".",config_name='train')
def main(cfg):
    env = hydra.utils.instantiate(cfg.env)
    # print(PACKAGE_PARENT)
    print(cfg.agent["name"])
    pass



if __name__ == '__main__':
    
    main()
    # env = FlexipodEnv(dof = 12)
    # utils.setSeedEverywhere(1)
    