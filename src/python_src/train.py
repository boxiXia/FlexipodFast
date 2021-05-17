import hydra
from pytorch_sac.train import Workspace
import torch

@hydra.main(config_path=".",config_name='train')
def main(cfg):
    
    print(torch.cuda.device_count())
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
