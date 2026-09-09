from __future__ import annotations

import torch

from .transformer import MachineZeroTransformer


def save_checkpoint(path,model,metadata:dict): torch.save({'model_state':model.state_dict(),'metadata':metadata},path)
def load_checkpoint(path,device='cpu'):
 d=torch.load(path,map_location=device); m=MachineZeroTransformer(**d['metadata']['model_config']); m.load_state_dict(d['model_state']); return m,d['metadata']
