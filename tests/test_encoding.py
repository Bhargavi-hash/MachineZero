from __future__ import annotations

import torch

from machinezero.aliencpu.generator import generate_architecture
from machinezero.aliencpu.instruction import Instruction
from machinezero.aliencpu.simulator import AlienCPU
from machinezero.aliencpu.state import CPUState
from machinezero.data.encoding import EVIDENCE_DIM, FEATURE_DIM, encode_transition, observable_evidence


def test_encoding_has_no_seed_and_correct_shape():
    s = CPUState([1, 2, 3, 4])
    x = encode_transition(s, Instruction(17, 0, 1), s, 8, 4)
    assert x.shape == (FEATURE_DIM,)
    assert FEATURE_DIM > EVIDENCE_DIM


def test_observable_evidence_uses_only_transition_and_contains_match():
    spec = generate_architecture(42)
    cpu = AlienCPU(spec)
    op = spec.opcodes[0]
    before = CPUState(list(range(spec.num_registers)))
    ins = Instruction(op.opcode, 0, 1)
    after = cpu.step(before, ins)
    evidence = observable_evidence(before, ins, after, spec.word_bits, spec.num_registers)
    assert len(evidence) == EVIDENCE_DIM
    assert any(v == 1.0 for v in evidence)
    assert torch.isfinite(torch.tensor(evidence)).all()
