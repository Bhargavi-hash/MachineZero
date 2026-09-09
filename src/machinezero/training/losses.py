import torch
from torch.nn import functional as F
def state_bit_loss(logits:torch.Tensor,target:torch.Tensor)->torch.Tensor: return F.binary_cross_entropy_with_logits(logits,target)
