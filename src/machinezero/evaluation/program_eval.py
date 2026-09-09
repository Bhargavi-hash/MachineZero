from __future__ import annotations

from machinezero.models.prediction import predict_state


def predict_program(model,context,state,program,word_bits,num_registers,device='cpu'):
 cur=state
 for ins in program: cur=predict_state(model,context,cur,ins,word_bits,num_registers,device); cur.pc += 1
 return cur
