from torch import nn, optim
import numpy as np
import torch

class FlexipodSimBaselineDataset(torch.utils.data.Dataset):
# class FlexipodSimBaselineDataset(torch.utils.data.IterableDataset):

    
    def __init__(self, 
                 state:np.ndarray, 
                 action:np.ndarray, 
                 state_next:np.ndarray, 
                 history_size: int = 1, 
                 batch_size: int = 1,
                 shuffle:bool=True,
                 verbose:bool = True):
        """
        Args:
            state: (np array), state
            action: (np array), action
            state_next: (np array), next state
        """
        self.state = torch.from_numpy(np.asarray(state, dtype=np.float32))
        self.action = torch.from_numpy(np.asarray(action, dtype=np.float32))
        self.state_next = torch.from_numpy(
            np.asarray(state_next, dtype=np.float32))
        self.history_size = history_size
        self.state_size = self.state.shape[1:]  # size of one state
        self.action_size = self.action.shape[1:]  # size of one action
        self.batch_size = batch_size
        self.len = int((len(self.state)-self.history_size)/self.batch_size)
        self.shuffle = shuffle
        if shuffle:
            self.indices = torch.randperm(len(self.state)-self.history_size)
        else:
            self.indices = torch.arange(len(self.state)-self.history_size)
        self.verbose=verbose
        # repeat tensor([0,1,..history_size-1]) batch_size times
        self.index_repeat = torch.arange(history_size).repeat(batch_size)

        self.state_batch_shape = torch.Size(
            [self.batch_size, self.history_size])+self.state_size
        self.action_batch_shape = torch.Size(
            [self.batch_size, self.history_size])+self.action_size

        self.idx = 0 # stat iteration index

    # custom memory pinning method

    def pin_memory(self):
        self.state = self.state.pin_memory()
        self.action = self.action.pin_memory()
        self.state_next = self.state_next.pin_memory()
        return self

    def __len__(self):
        # drop last batch if last batch is not enough
        return self.len

    def __getitem__(self, idx):
        # with torch.no_grad():
        # print(idx)
        # idx = idx%self.len
        start = self.batch_size*idx
        end = start+self.batch_size
#         print(start,end)
        # batch_indices = (id0,id1,...idn)
        batch_indices = self.indices[start:end]
        # batch_indices_repeat = 
        # (id0,id0+1,id0+2,...,id0+history_size,
        #  id1,id1+1,id1+2,...,id1+history_size,...,
        #  idn,idn+1,idn+2,...,idn+history_size)
        batch_indices_repeat = batch_indices.repeat_interleave(
            self.history_size)
        batch_indices_repeat+=self.index_repeat

        return {
            "s0": self.state[batch_indices_repeat].view(self.state_batch_shape),
            "a0": self.action[batch_indices_repeat].view(self.action_batch_shape),
            "s1": self.state_next[batch_indices_repeat].view(self.state_batch_shape)}
    
    def  __iter__(self):
        self.idx = 0 # stat iteration
        return self

    def __next__(self):
        # print(self.idx)
        if self.idx==self.len:
            self.idx = 0
            if self.shuffle:
                self.indices = torch.randperm(len(self.state)-self.history_size)
                if self.verbose:
                    print(f"random shuffle{self.indices[:10]}...")
        idx = self.idx
        self.idx+=1
        return self.__getitem__(idx)

        # while True:
        #     for idx in range(self.__len__()):
        #         yield self.__getitem__(idx)
        #     if self.shuffle:
        #         self.indices = torch.randperm(len(self.state)-self.history_size)

                    
            
