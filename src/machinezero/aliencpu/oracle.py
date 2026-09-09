from __future__ import annotations

from .instruction import Instruction
from .simulator import AlienCPU
from .state import CPUState


class HiddenOracle:
    '''Restricted black-box interface used by discovery agents.'''
    def __init__(self, cpu: AlienCPU): self.__cpu=cpu
    @property
    def word_bits(self)->int: return self.__cpu.spec.word_bits
    @property
    def num_registers(self)->int: return self.__cpu.spec.num_registers
    @property
    def valid_opcodes(self)->tuple[int,...]: return tuple(x.opcode for x in self.__cpu.spec.opcodes)
    def execute(self, initial_state: CPUState, instruction: Instruction)->CPUState:
        return self.__cpu.step(initial_state,instruction)
