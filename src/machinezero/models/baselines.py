from __future__ import annotations
import random
from machinezero.aliencpu.state import CPUState

class RandomStatePredictor:
    def __init__(self,word_bits:int,num_registers:int,seed:int=0):
        self.word_bits=word_bits; self.num_registers=num_registers; self.rng=random.Random(seed)
    def predict(self,*_args,**_kwargs)->CPUState:
        m=(1<<self.word_bits)-1
        return CPUState([self.rng.randint(0,m) for _ in range(self.num_registers)],self.rng.randint(0,1),self.rng.randint(0,1),0)

class NoOpPredictor:
    def __init__(self,word_bits:int): self.mask=(1<<word_bits)-1
    def predict(self,before:CPUState,*_args,**_kwargs)->CPUState:
        out=before.clone(); out.pc=(out.pc+1)&self.mask; return out
