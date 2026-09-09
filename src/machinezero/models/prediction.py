from __future__ import annotations
import torch
from machinezero.aliencpu.state import CPUState
from machinezero.aliencpu.instruction import Instruction
from machinezero.data.encoding import encode_transition,decode_target_bits

def predict_state(model,context:list[tuple[CPUState,Instruction,CPUState]],before:CPUState,ins:Instruction,word_bits:int,num_registers:int,device='cpu')->CPUState:
    toks=[encode_transition(a,b,c,word_bits,num_registers) for a,b,c in context]
    toks.append(encode_transition(before,ins,None,word_bits,num_registers,True)); x=torch.stack(toks)[None].to(device); mask=torch.zeros((1,len(toks)),dtype=torch.bool,device=device)
    model.eval()
    with torch.no_grad(): logits=model(x,mask)[0].cpu()
    return decode_target_bits(logits,word_bits,num_registers)
