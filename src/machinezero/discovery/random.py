from __future__ import annotations
import random
from .base import Explorer,DiscoveryResult
from machinezero.aliencpu.state import CPUState
from machinezero.aliencpu.instruction import Instruction
class RandomExplorer(Explorer):
    def discover(self,oracle,budget:int,seed:int=0)->DiscoveryResult:
        rng=random.Random(seed); out=[]; mask=(1<<oracle.word_bits)-1
        for _ in range(budget):
            s=CPUState([rng.randint(0,mask) for _ in range(oracle.num_registers)],rng.randint(0,1),rng.randint(0,1),0); op=rng.choice(oracle.valid_opcodes)
            ins=Instruction(op,rng.randrange(oracle.num_registers),rng.randint(0,max(mask,oracle.num_registers-1))); out.append((s,ins,oracle.execute(s,ins)))
        return DiscoveryResult(out)
