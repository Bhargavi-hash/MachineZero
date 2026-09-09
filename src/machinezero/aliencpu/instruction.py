from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Instruction:
    opcode: int
    a: int = 0
    b: int = 0

    def __str__(self) -> str:
        return f'{self.opcode:02X} {self.a:02X} {self.b:02X}'
