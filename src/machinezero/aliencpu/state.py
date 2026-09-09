from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CPUState:
    registers: list[int]
    zero: int = 0
    carry: int = 0
    pc: int = 0

    def clone(self) -> CPUState:
        return CPUState(self.registers.copy(), self.zero, self.carry, self.pc)
