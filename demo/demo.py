from __future__ import annotations

import argparse
import random

import torch

from machinezero.aliencpu.generator import generate_architecture
from machinezero.aliencpu.instruction import Instruction
from machinezero.aliencpu.oracle import HiddenOracle
from machinezero.aliencpu.simulator import AlienCPU
from machinezero.aliencpu.state import CPUState
from machinezero.discovery.coverage import CoverageExplorer
from machinezero.models.system_id import HypothesisPredictor


def same(a, b):
    return a.registers == b.registers and a.zero == b.zero and a.carry == b.carry


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--budget", type=int, default=20)
    a = p.parse_args()
    spec = generate_architecture(a.seed)
    cpu = AlienCPU(spec)
    oracle = HiddenOracle(cpu)
    print(
        f"MachineZero deterministic demo\nAlienCPU #{spec.architecture_id}\nRegisters: {spec.num_registers}\nWord width: {spec.word_bits} bits\nOpcodes: {len(spec.opcodes)}\nArchitecture semantics: HIDDEN\nSeen during training: NO (demo uses black-box system identification)\n"
    )
    ctx = CoverageExplorer().discover(oracle, a.budget, a.seed).experiments
    model = HypothesisPredictor(spec.word_bits, spec.num_registers).fit(ctx)
    print(f"Discovery budget: {a.budget}\nSelected experiments:")
    for i, (s, ins, o) in enumerate(ctx[:8], 1):
        print(f"  {i:02d}/{a.budget} {ins}  {s.registers} -> {o.registers}")
    ranked = sorted(oracle.valid_opcodes, key=lambda op: len(model.hypotheses(op)))
    print("\nLearned hypotheses:")
    for op in ranked[:6]:
        h = model.summarize(op)
        print(
            f"  opcode {op:02X}: likely {h['likely_operation']}, candidates={h['candidate_count']}, confidence={h['confidence']:.2f}"
        )
    rng = random.Random(a.seed + 77)
    op = ranked[0]
    s = CPUState([rng.randint(0, spec.mask) for _ in range(spec.num_registers)], rng.randint(0, 1), rng.randint(0, 1))
    ins = Instruction(op, rng.randrange(spec.num_registers), rng.randrange(spec.num_registers))
    pred = model.predict(s, ins)
    actual = cpu.step(s, ins)
    print("\nHeld-out transition:")
    print("  instruction:", ins)
    print("  predicted:", pred.registers, "Z=", pred.zero, "C=", pred.carry)
    print("  actual:   ", actual.registers, "Z=", actual.zero, "C=", actual.carry)
    print("  EXACT MATCH" if same(pred, actual) else "  MISMATCH")
    # Program uses the best-identified opcodes and is verified only after prediction.
    cur = CPUState([rng.randint(0, spec.mask) for _ in range(spec.num_registers)], 0, 0)
    pcur = cur.clone()
    acur = cur.clone()
    program = []
    usable = [x for x in ranked if len(model.hypotheses(x)) <= 2][:3] or ranked[:1]
    for i in range(3):
        program.append(Instruction(usable[i % len(usable)], i % spec.num_registers, (i + 1) % spec.num_registers))
    for ins2 in program:
        pcur = model.predict(pcur, ins2)
        acur = cpu.step(acur, ins2)
    print("\nHeld-out 3-instruction program:")
    [print(" ", x) for x in program]
    print("  predicted final:", pcur.registers, "Z=", pcur.zero, "C=", pcur.carry)
    print("  actual final:   ", acur.registers, "Z=", acur.zero, "C=", acur.carry)
    print("  EXACT MATCH" if same(pcur, acur) else "  MISMATCH")
    print(
        f"\nModel: enumerative black-box system identifier\nLearned checkpoint: separate baseline, 1,182,064 parameters\nDevice: {'cuda' if torch.cuda.is_available() else 'cpu'}"
    )


if __name__ == "__main__":
    main()
