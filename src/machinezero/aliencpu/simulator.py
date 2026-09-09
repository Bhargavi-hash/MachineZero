from __future__ import annotations
from .architecture import ArchitectureSpec, OpcodeSpec
from .instruction import Instruction
from .state import CPUState

class AlienCPU:
    def __init__(self, spec: ArchitectureSpec):
        self.spec=spec
        self._map=spec.opcode_map()
        self.state=self.reset()

    def reset(self, state: CPUState|None=None) -> CPUState:
        self.state = state.clone() if state else CPUState([0]*self.spec.num_registers)
        return self.state.clone()

    def clone_state(self) -> CPUState:
        return self.state.clone()

    def validate_instruction(self, ins: Instruction) -> bool:
        return ins.opcode in self._map

    def _regs(self, s: OpcodeSpec, ins: Instruction) -> tuple[int,int]:
        a=ins.a % self.spec.num_registers; b=ins.b % self.spec.num_registers
        return (a,b) if s.dst_first else (b,a)

    def step(self, state: CPUState, ins: Instruction) -> CPUState:
        if ins.opcode not in self._map: raise ValueError(f'Unknown opcode {ins.opcode}')
        out=state.clone(); sp=self._map[ins.opcode]; m=self.spec.mask
        dst,src=self._regs(sp,ins)
        old=out.registers[dst]; rhs=out.registers[src]
        result=None; carry=out.carry
        op=sp.operation
        if op=='MOV': result=rhs
        elif op=='ADD':
            raw=old+rhs; result=raw&m; carry=int(raw>m)
        elif op=='SUB':
            result=(old-rhs)&m; carry=int(old>=rhs)
        elif op=='XOR': result=old^rhs
        elif op=='AND': result=old&rhs
        elif op=='OR': result=old|rhs
        elif op=='NOT': result=(~old)&m
        elif op=='SHL': carry=(old>>(self.spec.word_bits-1))&1; result=(old<<1)&m
        elif op=='SHR': carry=old&1; result=(old>>1)&m
        elif op=='LOAD_IMMEDIATE': result=ins.b&m; dst=ins.a%self.spec.num_registers
        elif op=='COMPARE':
            result=(old-rhs)&m; carry=int(old>=rhs)
        elif op=='JMP': out.pc=ins.a&m
        elif op=='JZ':
            if out.zero: out.pc=ins.a&m
        if result is not None and op!='COMPARE': out.registers[dst]=result&m
        if sp.updates_zero and result is not None: out.zero=int((result&m)==0)
        if sp.updates_carry: out.carry=carry
        if op not in {'JMP','JZ'} or out.pc==state.pc: out.pc=(state.pc+1)&m
        return out

    def run(self, program: list[Instruction], state: CPUState|None=None, max_steps:int|None=None) -> CPUState:
        cur=state.clone() if state else CPUState([0]*self.spec.num_registers)
        max_steps=max_steps or max(1,len(program)*4)
        steps=0
        while 0 <= cur.pc < len(program) and steps<max_steps:
            cur=self.step(cur,program[cur.pc]); steps+=1
        return cur
