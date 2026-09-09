import torch
from machinezero.models.transformer import MachineZeroTransformer
def test_model_shape():
 m=MachineZeroTransformer(d_model=32,nhead=4,num_layers=1,dim_feedforward=64,max_tokens=6); y=m(torch.randn(2,6,28),torch.zeros(2,6,dtype=torch.bool)); assert y.shape==(2,130)
