from __future__ import annotations

import torch

from machinezero.aliencpu.architecture import OpcodeSpec
from machinezero.aliencpu.instruction import Instruction
from machinezero.aliencpu.state import CPUState

MAX_REGS = 8
MAX_BITS = 16
OPCODE_VOCAB = 256

# before regs 8 + after regs 8 + before/after flags 4 + before/after pc 2
# + opcode one-hot 256 + operands 2 + word bits + num regs + query marker = 283.
# The opcode is deliberately represented as an identity, not a magnitude: opcode 0x10
# is not semantically "closer" to 0x11 than to 0xF0.
FEATURE_DIM = 2 * MAX_REGS + 4 + 2 + OPCODE_VOCAB + 2 + 4

OPERATIONS = (
    'MOV',
    'ADD',
    'SUB',
    'XOR',
    'AND',
    'OR',
    'NOT',
    'SHL',
    'SHR',
    'LOAD_IMMEDIATE',
    'COMPARE',
    'JMP',
    'JZ',
)
OP_TO_INDEX = {op: i for i, op in enumerate(OPERATIONS)}
SEMANTIC_DIM = len(OPERATIONS) + 3  # operation logits + dst_first/update_zero/update_carry


def _norm(v: int, mask: int) -> float:
    return float(v) / float(max(mask, 1))


def encode_transition(
    before: CPUState,
    ins: Instruction,
    after: CPUState | None,
    word_bits: int,
    num_registers: int,
    is_query: bool = False,
    query_opcode: int | None = None,
) -> torch.Tensor:
    """Encode one observed transition without architecture IDs or hidden semantics."""
    mask = (1 << word_bits) - 1
    x: list[float] = []
    x += [_norm(before.registers[i], mask) if i < num_registers else 0.0 for i in range(MAX_REGS)]
    x += [
        _norm(after.registers[i], mask) if after is not None and i < num_registers else 0.0
        for i in range(MAX_REGS)
    ]
    x += [
        float(before.zero),
        float(before.carry),
        float(after.zero) if after else 0.0,
        float(after.carry) if after else 0.0,
    ]
    x += [_norm(before.pc, mask), _norm(after.pc, mask) if after else 0.0]

    opcode = [0.0] * OPCODE_VOCAB
    opcode[ins.opcode & 0xFF] = 1.0
    x += opcode
    same_query_opcode = float(query_opcode is not None and ins.opcode == query_opcode)
    x += [
        ins.a / 255.0,
        ins.b / 255.0,
        word_bits / 16.0,
        num_registers / MAX_REGS,
        float(is_query),
        same_query_opcode,
    ]
    assert len(x) == FEATURE_DIM
    return torch.tensor(x, dtype=torch.float32)


def semantic_target(spec: OpcodeSpec) -> torch.Tensor:
    """Training-only target describing semantics; never included in model inputs."""
    return torch.tensor(
        [
            OP_TO_INDEX[spec.operation],
            int(spec.dst_first),
            int(spec.updates_zero),
            int(spec.updates_carry),
        ],
        dtype=torch.long,
    )


def decode_semantics(logits: torch.Tensor) -> tuple[str, bool, bool, bool]:
    """Decode the model's latent ISA hypothesis for one query opcode."""
    n_ops = len(OPERATIONS)
    op = OPERATIONS[int(logits[:n_ops].argmax().item())]
    attrs = logits[n_ops : n_ops + 3].sigmoid() >= 0.5
    return op, bool(attrs[0].item()), bool(attrs[1].item()), bool(attrs[2].item())
