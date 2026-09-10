from __future__ import annotations

import torch

from machinezero.aliencpu.instruction import Instruction
from machinezero.aliencpu.state import CPUState
from machinezero.data.encoding import decode_semantics, encode_transition


def execute_hypothesis(
    before: CPUState,
    ins: Instruction,
    word_bits: int,
    num_registers: int,
    operation: str,
    dst_first: bool,
    updates_zero: bool,
    updates_carry: bool,
) -> CPUState:
    """Execute one instruction using only model-inferred semantics."""
    out = before.clone()
    mask = (1 << word_bits) - 1
    a, b = ins.a % num_registers, ins.b % num_registers
    dst, src = (a, b) if dst_first else (b, a)
    old, rhs = out.registers[dst], out.registers[src]
    result: int | None = None
    carry = out.carry

    if operation == "MOV":
        result = rhs
    elif operation == "ADD":
        raw = old + rhs
        result, carry = raw & mask, int(raw > mask)
    elif operation == "SUB":
        result, carry = (old - rhs) & mask, int(old >= rhs)
    elif operation == "XOR":
        result = old ^ rhs
    elif operation == "AND":
        result = old & rhs
    elif operation == "OR":
        result = old | rhs
    elif operation == "NOT":
        result = (~old) & mask
    elif operation == "SHL":
        carry, result = (old >> (word_bits - 1)) & 1, (old << 1) & mask
    elif operation == "SHR":
        carry, result = old & 1, old >> 1
    elif operation == "LOAD_IMMEDIATE":
        dst, result = ins.a % num_registers, ins.b & mask
    elif operation == "COMPARE":
        result, carry = (old - rhs) & mask, int(old >= rhs)
    elif operation == "JMP" or operation == "JZ" and out.zero:
        out.pc = ins.a & mask

    if result is not None and operation != "COMPARE":
        out.registers[dst] = result & mask
    if updates_zero and result is not None:
        out.zero = int((result & mask) == 0)
    if updates_carry:
        out.carry = carry
    if operation not in {"JMP", "JZ"} or out.pc == before.pc:
        out.pc = (before.pc + 1) & mask
    return out


def infer_semantics(
    model,
    context: list[tuple[CPUState, Instruction, CPUState]],
    before: CPUState,
    ins: Instruction,
    word_bits: int,
    num_registers: int,
    device: str = "cpu",
) -> tuple[str, bool, bool, bool]:
    """Infer a query opcode's semantics from black-box context transitions."""
    tokens = [encode_transition(a, b, c, word_bits, num_registers, query_opcode=ins.opcode) for a, b, c in context]
    tokens.append(
        encode_transition(
            before,
            ins,
            None,
            word_bits,
            num_registers,
            True,
            query_opcode=ins.opcode,
        )
    )
    x = torch.stack(tokens)[None].to(device)
    padding_mask = torch.zeros((1, len(tokens)), dtype=torch.bool, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(x, padding_mask)[0].cpu()
    return decode_semantics(logits)


def predict_state(
    model,
    context: list[tuple[CPUState, Instruction, CPUState]],
    before: CPUState,
    ins: Instruction,
    word_bits: int,
    num_registers: int,
    device: str = "cpu",
) -> CPUState:
    semantics = infer_semantics(model, context, before, ins, word_bits, num_registers, device)
    return execute_hypothesis(before, ins, word_bits, num_registers, *semantics)
