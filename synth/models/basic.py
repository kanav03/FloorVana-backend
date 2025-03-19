from collections import OrderedDict
import torch as t
import os
import re

class BasicModule(t.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.name = str(type(self))

    def load_model(self, path, from_multi_GPU=False):
        state_dict = t.load(path, map_location=t.device('cpu'))
        if from_multi_GPU:
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:]  # remove 'module.' prefix
                new_state_dict[name] = v
            state_dict = new_state_dict
        self.load_state_dict(state_dict)

    def save_model(self, epoch=0):  
        pth_list = [pth for pth in os.listdir('checkpoints') if re.match(self.name, pth)]
        pth_list = sorted(pth_list, key=lambda x: os.path.getmtime(os.path.join('checkpoints', x)))
        if len(pth_list) >= 10 and pth_list is not None:
            to_delete = 'checkpoints/' + pth_list[0]
            if os.path.exists(to_delete):
                os.remove(to_delete)  
                
        path = f'checkpoints/{self.name}_{epoch}.pth'
        t.save(self.state_dict(), path)

class ParallelModule():
    def __init__(self, model, device_ids=[0, 1]):
        self.name = model.name
        self.model = t.nn.DataParallel(model, device_ids=device_ids) 

    def load_model(self, path, from_multi_GPU=True):
        state_dict = t.load(path, map_location=t.device('cpu'))
        if from_multi_GPU:
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:]  # remove 'module.' prefix
                new_state_dict[name] = v
            state_dict = new_state_dict
        self.model.load_state_dict(state_dict)

    def save_model(self, epoch=0):  
        pth_list = [pth for pth in os.listdir('checkpoints') if re.match(self.name, pth)]
        pth_list = sorted(pth_list, key=lambda x: os.path.getmtime(os.path.join('checkpoints', x)))
        if len(pth_list) >= 10 and pth_list is not None:
            to_delete = 'checkpoints/' + pth_list[0]
            if os.path.exists(to_delete):
                os.remove(to_delete)  
                
        path = f'checkpoints/{self.name}_parallel_{epoch}.pth'
        t.save(self.model.state_dict(), path)