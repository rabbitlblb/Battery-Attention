import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.modules.container import ModuleList

from utils import to_var
from model.tasks import to_array, to_tensor
import numpy as np
from model import dynamic_vae

class DynamicVAE(nn.Module):

    def __init__(self, model, **params):
        super().__init__()
        self.models = []
        for i in range(params['battery_num']):
            self.models.append(model.DynamicVAE(**params))
        self.models = ModuleList(self.models)
        print(self.models)
    

    def forward(self, input_sequence, **params):
        '''
        rets = []
        car = to_var(params['car']).float().unsqueeze(1)
        for i in range(car.shape[0]):
            ret = self.models[i](input_sequence, **params)
            for j in range(len(ret)):
                if len(rets) <= j:
                    rets.append([])
                rets[j].append(ret[j])
        for i in range(len(rets)):
            rets[i] = torch.matmul(torch.stack(rets[i], dim=-1), car).squeeze(-1)
            
        return rets
        '''
        battery_num = params['car'].argmax()
        return self.models[battery_num].forward(input_sequence, **params)
