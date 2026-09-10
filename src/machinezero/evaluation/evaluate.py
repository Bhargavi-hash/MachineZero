from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from machinezero.aliencpu.generator import generate_architecture
from machinezero.aliencpu.instruction import Instruction
from machinezero.aliencpu.oracle import HiddenOracle
from machinezero.aliencpu.simulator import AlienCPU
from machinezero.aliencpu.state import CPUState
from machinezero.data.splits import make_splits
from machinezero.discovery.active import ModelExplorer
from machinezero.discovery.coverage import CoverageExplorer
from machinezero.discovery.random import RandomExplorer
from machinezero.models.checkpoint import load_checkpoint
from machinezero.models.prediction import execute_hypothesis, infer_semantics

from .metrics import compare_states


def sample_query(rng, spec):
    state = CPUState(
        [rng.randint(0, spec.mask) for _ in range(spec.num_registers)],
        rng.randint(0, 1),
        rng.randint(0, 1),
        0,
    )
    opcode_spec = rng.choice(spec.opcodes)
    if opcode_spec.operation == "LOAD_IMMEDIATE":
        ins = Instruction(opcode_spec.opcode, rng.randrange(spec.num_registers), rng.randint(0, spec.mask))
    elif opcode_spec.operation in {"JMP", "JZ"}:
        ins = Instruction(opcode_spec.opcode, rng.randint(0, 15), 0)
    else:
        ins = Instruction(
            opcode_spec.opcode,
            rng.randrange(spec.num_registers),
            rng.randrange(spec.num_registers),
        )
    return state, ins


def evaluate_checkpoint(
    path: str,
    budgets=(0, 1, 2, 5, 10, 20, 30),
    queries_per_arch: int = 12,
    include_active: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_checkpoint(path, device)
    dc = meta["data_config"]
    splits = make_splits(
        dc["base_seed"],
        dc["train_architectures"],
        dc["validation_architectures"],
        dc["test_architectures"],
    )
    explorers = {"random": RandomExplorer(), "coverage": CoverageExplorer()}
    if include_active:
        explorers["model"] = ModelExplorer(model, device=device)

    rows = []
    for name, explorer in explorers.items():
        for budget in budgets:
            sums = {
                "exact": 0.0,
                "register_accuracy": 0.0,
                "flag_accuracy": 0.0,
                "operation_accuracy": 0.0,
                "semantic_exact": 0.0,
            }
            observed_exact = 0.0
            observed_count = 0
            unobserved_exact = 0.0
            unobserved_count = 0
            count = 0

            for arch_seed in splits.test:
                spec = generate_architecture(arch_seed)
                cpu = AlienCPU(spec)
                oracle = HiddenOracle(cpu)
                context = explorer.discover(oracle, budget, seed=arch_seed + budget).experiments if budget else []
                rng = random.Random(arch_seed * 31 + budget)
                opcode_map = spec.opcode_map()

                for _ in range(queries_per_arch):
                    state, ins = sample_query(rng, spec)
                    actual = cpu.step(state, ins)
                    inferred = infer_semantics(
                        model,
                        context,
                        state,
                        ins,
                        spec.word_bits,
                        spec.num_registers,
                        device,
                    )
                    pred = execute_hypothesis(
                        state,
                        ins,
                        spec.word_bits,
                        spec.num_registers,
                        *inferred,
                    )
                    metrics = compare_states(pred, actual)
                    truth = opcode_map[ins.opcode]
                    expected = (
                        truth.operation,
                        truth.dst_first,
                        truth.updates_zero,
                        truth.updates_carry,
                    )
                    metrics["operation_accuracy"] = float(inferred[0] == truth.operation)
                    metrics["semantic_exact"] = float(inferred == expected)
                    for key in sums:
                        sums[key] += metrics[key]

                    seen = any(ctx_ins.opcode == ins.opcode for _, ctx_ins, _ in context)
                    if seen:
                        observed_count += 1
                        observed_exact += metrics["exact"]
                    else:
                        unobserved_count += 1
                        unobserved_exact += metrics["exact"]
                    count += 1

            row = {
                "explorer": name,
                "budget": budget,
                "architectures": len(splits.test),
                "queries": count,
                **{key: value / count for key, value in sums.items()},
                "query_opcode_observed_fraction": observed_count / count,
                "exact_when_opcode_observed": observed_exact / observed_count if observed_count else None,
                "exact_when_opcode_unobserved": (unobserved_exact / unobserved_count if unobserved_count else None),
            }
            rows.append(row)
            print(row)

    return {
        "checkpoint": path,
        "device": device,
        "held_out_test_seeds": list(splits.test),
        "results": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="results/eval.json")
    parser.add_argument("--active", action="store_true")
    parser.add_argument("--queries-per-arch", type=int, default=12)
    args = parser.parse_args()
    results = evaluate_checkpoint(
        args.checkpoint,
        queries_per_arch=args.queries_per_arch,
        include_active=args.active,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
