from __future__ import annotations

import random

from .architecture import ArchitectureSpec, OpcodeSpec, architecture_id_from_payload

OPS = {
    "MOV": 2,
    "ADD": 2,
    "SUB": 2,
    "XOR": 2,
    "AND": 2,
    "OR": 2,
    "NOT": 1,
    "SHL": 1,
    "SHR": 1,
    "LOAD_IMMEDIATE": 2,
    "COMPARE": 2,
    "JMP": 1,
    "JZ": 1,
}
MANDATORY = ["MOV", "ADD", "SUB", "XOR", "LOAD_IMMEDIATE"]
OPTIONAL = ["AND", "OR", "NOT", "SHL", "SHR", "COMPARE", "JMP", "JZ"]


def generate_architecture(seed: int) -> ArchitectureSpec:
    rng = random.Random(seed)
    word_bits = rng.choice([8, 12, 16])
    num_registers = rng.randint(4, 8)
    has_zero = rng.random() < 0.85
    has_carry = rng.random() < 0.55
    target_n = rng.randint(8, min(13, len(MANDATORY) + len(OPTIONAL)))
    chosen = MANDATORY + rng.sample(OPTIONAL, target_n - len(MANDATORY))
    opcode_values = rng.sample(range(1, 256), len(chosen))
    specs = []
    for op, code in zip(chosen, opcode_values):
        specs.append(
            OpcodeSpec(
                opcode=code,
                operation=op,
                arity=OPS[op],
                dst_first=(rng.random() < 0.5 if OPS[op] == 2 and op not in {"LOAD_IMMEDIATE", "JMP", "JZ"} else True),
                updates_zero=has_zero
                and op in {"ADD", "SUB", "XOR", "AND", "OR", "NOT", "SHL", "SHR", "MOV", "LOAD_IMMEDIATE"}
                and rng.random() < 0.8,
                updates_carry=has_carry and op in {"ADD", "SUB", "SHL", "SHR"} and rng.random() < 0.75,
            )
        )
    base = {
        "seed": seed,
        "word_bits": word_bits,
        "num_registers": num_registers,
        "memory_size": 0,
        "instruction_width": 3,
        "has_zero_flag": has_zero,
        "has_carry_flag": has_carry,
        "opcodes": tuple(specs),
    }
    payload = {**base, "opcodes": [s.__dict__ for s in specs]}
    return ArchitectureSpec(architecture_id=architecture_id_from_payload(payload), **base)
