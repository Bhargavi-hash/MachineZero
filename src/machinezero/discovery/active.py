from __future__ import annotations

import random

import torch

from machinezero.aliencpu.instruction import Instruction
from machinezero.aliencpu.state import CPUState
from machinezero.data.encoding import encode_transition

from .base import DiscoveryResult, Explorer


class ModelExplorer(Explorer):
    def __init__(self,model,device='cpu',candidates:int=24,mc_samples:int=5): self.model=model; self.device=device; self.candidates=candidates; self.mc_samples=mc_samples
    def discover(self,oracle,budget:int,seed:int=0)->DiscoveryResult:
        rng=random.Random(seed); ctx=[]; maskv=(1<<oracle.word_bits)-1
        for _ in range(budget):
            cand=[]
            for _ in range(self.candidates):
                s=CPUState([rng.randint(0,maskv) for _ in range(oracle.num_registers)],rng.randint(0,1),rng.randint(0,1),0); ins=Instruction(rng.choice(oracle.valid_opcodes),rng.randrange(oracle.num_registers),rng.randrange(oracle.num_registers)); cand.append((s,ins))
            scores=[]; self.model.train()
            for s,ins in cand:
                toks=[encode_transition(a,b,c,oracle.word_bits,oracle.num_registers) for a,b,c in ctx]+[encode_transition(s,ins,None,oracle.word_bits,oracle.num_registers,True)]
                x=torch.stack(toks)[None].to(self.device); pm=torch.zeros((1,len(toks)),dtype=torch.bool,device=self.device); preds=[]
                with torch.no_grad():
                    for _ in range(self.mc_samples): preds.append(self.model(x,pm).sigmoid())
                scores.append(torch.stack(preds).var(0).mean().item())
            s,ins=cand[max(range(len(scores)),key=scores.__getitem__)]; ctx.append((s,ins,oracle.execute(s,ins)))
        self.model.eval(); return DiscoveryResult(ctx)
