from __future__ import annotations
import torch
from torch import nn
from machinezero.data.encoding import FEATURE_DIM,MAX_REGS,MAX_BITS

class MachineZeroTransformer(nn.Module):
    def __init__(self,d_model:int=96,nhead:int=4,num_layers:int=3,dim_feedforward:int=192,dropout:float=.1,max_tokens:int=31):
        super().__init__(); self.d_model=d_model
        self.input=nn.Sequential(nn.Linear(FEATURE_DIM,d_model),nn.GELU(),nn.LayerNorm(d_model))
        self.pos=nn.Parameter(torch.randn(1,max_tokens,d_model)*.02)
        layer=nn.TransformerEncoderLayer(d_model,nhead,dim_feedforward,dropout,batch_first=True,norm_first=True,activation='gelu')
        self.encoder=nn.TransformerEncoder(layer,num_layers)
        self.head=nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,MAX_REGS*MAX_BITS+2))
    def forward(self,tokens:torch.Tensor,padding_mask:torch.Tensor|None=None)->torch.Tensor:
        x=self.input(tokens)+self.pos[:,:tokens.shape[1]]; h=self.encoder(x,src_key_padding_mask=padding_mask)
        if padding_mask is None: q=h[:,-1]
        else:
            idx=(~padding_mask).sum(1)-1; q=h[torch.arange(h.size(0),device=h.device),idx]
        return self.head(q)
    def parameter_count(self)->int: return sum(p.numel() for p in self.parameters())
