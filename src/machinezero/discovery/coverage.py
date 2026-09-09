from __future__ import annotations
from .base import Explorer,DiscoveryResult
from machinezero.aliencpu.state import CPUState
from machinezero.aliencpu.instruction import Instruction

class CoverageExplorer(Explorer):
    '''Deterministic probes designed to separate common ALU hypotheses.'''
    def discover(self,oracle,budget:int,seed:int=0)->DiscoveryResult:
        out=[]; mask=(1<<oracle.word_bits)-1; ops=list(oracle.valid_opcodes)
        # Each pass uses a distinct pair. 3/5 separates MOV, ADD, SUB, XOR, AND, OR;
        # max/1 exposes wrapping/carry; 0/0 exposes NOT/shift/flags.
        pairs=[(3,5),(mask,1),(0,0),(2,1),(1<<(oracle.word_bits-1),1)]
        for i in range(budget):
            op=ops[i%len(ops)]; pass_i=i//len(ops); va,vb=pairs[pass_i%len(pairs)]
            a=i%oracle.num_registers; b=(i+1)%oracle.num_registers
            regs=[0]*oracle.num_registers; regs[a]=va; regs[b]=vb
            s=CPUState(regs,zero=pass_i%2,carry=(pass_i//2)%2,pc=0)
            # b also acts as an immediate for LOAD_IMMEDIATE; use a useful immediate on later passes.
            operand_b = b if pass_i==0 else (vb & 0xFF)
            ins=Instruction(op,a,operand_b)
            out.append((s,ins,oracle.execute(s,ins)))
        return DiscoveryResult(out)
