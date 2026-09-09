from __future__ import annotations
from collections import Counter
from dataclasses import replace
from machinezero.aliencpu.architecture import ArchitectureSpec, OpcodeSpec
from machinezero.aliencpu.instruction import Instruction
from machinezero.aliencpu.state import CPUState
from machinezero.aliencpu.simulator import AlienCPU

OPS_ARITY={'MOV':2,'ADD':2,'SUB':2,'XOR':2,'AND':2,'OR':2,'NOT':1,'SHL':1,'SHR':1,'LOAD_IMMEDIATE':2,'COMPARE':2,'JMP':1,'JZ':1}

class HypothesisPredictor:
    '''Enumerative system-identification model using only black-box transitions.'''
    def __init__(self,word_bits:int,num_registers:int):
        self.word_bits=word_bits; self.num_registers=num_registers
        self.context=[]

    def fit(self,context): self.context=list(context); return self

    def _candidate_specs(self,opcode:int):
        for op,arity in OPS_ARITY.items():
            dst_opts=[True,False] if arity==2 and op not in {'LOAD_IMMEDIATE'} else [True]
            for dst_first in dst_opts:
                for upz in [False,True]:
                    for upc in [False,True]:
                        yield OpcodeSpec(opcode,op,arity,dst_first,upz,upc)

    def _cpu(self,cand:OpcodeSpec)->AlienCPU:
        spec=ArchitectureSpec('hypothesis',0,self.word_bits,self.num_registers,0,3,True,True,(cand,))
        return AlienCPU(spec)

    @staticmethod
    def _same(a:CPUState,b:CPUState)->bool:
        return a.registers==b.registers and a.zero==b.zero and a.carry==b.carry and a.pc==b.pc

    def hypotheses(self,opcode:int)->list[OpcodeSpec]:
        obs=[x for x in self.context if x[1].opcode==opcode]
        if not obs: return list(self._candidate_specs(opcode))
        valid=[]
        for cand in self._candidate_specs(opcode):
            cpu=self._cpu(cand)
            if all(self._same(cpu.step(before,ins),after) for before,ins,after in obs): valid.append(cand)
        return valid

    def predict(self,before:CPUState,ins:Instruction)->CPUState:
        hs=self.hypotheses(ins.opcode)
        if not hs:
            out=before.clone(); out.pc=(out.pc+1)&((1<<self.word_bits)-1); return out
        predictions=[self._cpu(h).step(before,ins) for h in hs]
        keys=[(tuple(p.registers),p.zero,p.carry,p.pc) for p in predictions]
        best,_=Counter(keys).most_common(1)[0]
        return CPUState(list(best[0]),best[1],best[2],best[3])

    def summarize(self,opcode:int)->dict:
        hs=self.hypotheses(opcode); ops=Counter(h.operation for h in hs)
        total=max(1,sum(ops.values())); best,count=ops.most_common(1)[0] if ops else ('UNKNOWN',0)
        return {'opcode':opcode,'candidate_count':len(hs),'likely_operation':best,'confidence':count/total}
