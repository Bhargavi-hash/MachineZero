from __future__ import annotations
from machinezero.aliencpu.state import CPUState
def compare_states(pred:CPUState,actual:CPUState)->dict[str,float]:
 n=len(actual.registers); reg=sum(int(a==b) for a,b in zip(pred.registers,actual.registers))/n
 exact=float(pred.registers==actual.registers and pred.zero==actual.zero and pred.carry==actual.carry)
 return {'exact':exact,'register_accuracy':reg,'flag_accuracy':(int(pred.zero==actual.zero)+int(pred.carry==actual.carry))/2}
