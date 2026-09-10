from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

Operation = Literal[
    "MOV", "ADD", "SUB", "XOR", "AND", "OR", "NOT", "SHL", "SHR", "LOAD_IMMEDIATE", "COMPARE", "JMP", "JZ"
]


@dataclass(frozen=True)
class OpcodeSpec:
    opcode: int
    operation: Operation
    arity: int
    dst_first: bool = True
    updates_zero: bool = False
    updates_carry: bool = False


@dataclass(frozen=True)
class ArchitectureSpec:
    architecture_id: str
    seed: int
    word_bits: int
    num_registers: int
    memory_size: int
    instruction_width: int
    has_zero_flag: bool
    has_carry_flag: bool
    opcodes: tuple[OpcodeSpec, ...]

    @property
    def mask(self) -> int:
        return (1 << self.word_bits) - 1

    def opcode_map(self) -> dict[int, OpcodeSpec]:
        return {x.opcode: x for x in self.opcodes}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["opcodes"] = [asdict(x) for x in self.opcodes]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def architecture_id_from_payload(payload: dict) -> str:
    safe = {k: v for k, v in payload.items() if k not in {"architecture_id"}}
    blob = json.dumps(safe, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:6]
