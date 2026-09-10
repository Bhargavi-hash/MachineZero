from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from machinezero.aliencpu.generator import generate_architecture
from machinezero.aliencpu.instruction import Instruction
from machinezero.aliencpu.oracle import HiddenOracle
from machinezero.aliencpu.simulator import AlienCPU
from machinezero.aliencpu.state import CPUState
from machinezero.discovery.coverage import CoverageExplorer
from machinezero.discovery.random import RandomExplorer
from machinezero.models.system_id import HypothesisPredictor

PROGRAM_ARGUMENT = typer.Argument(None, exists=True, dir_okay=False)

app = typer.Typer(
    no_args_is_help=True,
    help="MachineZero: learning to understand computers never seen before.",
)
console = Console()


def _summary(spec, reveal: bool = False):
    console.print(f"[bold]AlienCPU #{spec.architecture_id}[/bold]")
    console.print(
        f"Registers: {spec.num_registers}\n"
        f"Word width: {spec.word_bits} bits\n"
        f"Opcodes: {len(spec.opcodes)}\n"
        f"Architecture semantics: {'REVEALED' if reveal else 'HIDDEN'}"
    )
    if reveal:
        table = Table("Opcode", "Operation", "Dst first", "Z", "C")
        for item in spec.opcodes:
            table.add_row(
                f"{item.opcode:02X}",
                item.operation,
                str(item.dst_first),
                str(item.updates_zero),
                str(item.updates_carry),
            )
        console.print(table)


def _load_program(path: Path) -> list[Instruction]:
    program = []
    for line_no, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3:
            raise typer.BadParameter(f'{path}:{line_no}: expected "OPCODE A B"')
        try:
            opcode, a, b = (int(part, 16) for part in parts)
        except ValueError as exc:
            raise typer.BadParameter(f"{path}:{line_no}: operands must be hexadecimal bytes") from exc
        program.append(Instruction(opcode, a, b))
    if not program:
        raise typer.BadParameter(f"{path}: program contains no instructions")
    return program


@app.command()
def info():
    console.print(
        "[bold]MachineZero[/bold]\nLearning to understand computers never seen before.\nCore environment: AlienCPU"
    )


@app.command()
def generate(seed: int = typer.Option(42)):
    _summary(generate_architecture(seed), False)


@app.command()
def inspect(
    seed: int = typer.Option(42),
    reveal: bool = typer.Option(False, "--reveal"),
):
    _summary(generate_architecture(seed), reveal)


@app.command()
def discover(
    seed: int = typer.Option(42),
    budget: int = typer.Option(20),
    explorer: str = typer.Option("coverage"),
):
    spec = generate_architecture(seed)
    oracle = HiddenOracle(AlienCPU(spec))
    if explorer == "coverage":
        explorer_impl = CoverageExplorer()
    elif explorer == "random":
        explorer_impl = RandomExplorer()
    else:
        raise typer.BadParameter(
            "discover supports --explorer coverage or random; model exploration requires a checkpoint during evaluation"
        )

    context = explorer_impl.discover(oracle, budget, seed).experiments
    model = HypothesisPredictor(spec.word_bits, spec.num_registers).fit(context)
    _summary(spec, False)
    console.print(f"\n[bold]Experiments: {budget} ({explorer})[/bold]")
    for i, (before, ins, after) in enumerate(context[: min(8, len(context))], 1):
        console.print(f"{i:02d}. {ins}  {before.registers} -> {after.registers}")

    table = Table("Opcode", "Likely operation", "Candidates", "Confidence")
    for opcode in oracle.valid_opcodes:
        hypothesis = model.summarize(opcode)
        table.add_row(
            f"{opcode:02X}",
            hypothesis["likely_operation"],
            str(hypothesis["candidate_count"]),
            f"{hypothesis['confidence']:.2f}",
        )
    console.print("\n[bold]Learned hypotheses[/bold]")
    console.print(table)


@app.command()
def predict(
    program: Path | None = PROGRAM_ARGUMENT,
    seed: int = typer.Option(42),
    budget: int = typer.Option(20),
):
    """Predict a held-out transition or a hexadecimal .mz instruction sequence."""
    spec = generate_architecture(seed)
    cpu = AlienCPU(spec)
    oracle = HiddenOracle(cpu)
    context = CoverageExplorer().discover(oracle, budget, seed).experiments
    model = HypothesisPredictor(spec.word_bits, spec.num_registers).fit(context)
    rng = random.Random(seed + 999)

    if program is None:
        opcode_spec = rng.choice(spec.opcodes)
        state = CPUState(
            [rng.randint(0, spec.mask) for _ in range(spec.num_registers)],
            rng.randint(0, 1),
            rng.randint(0, 1),
        )
        ins = Instruction(
            opcode_spec.opcode,
            rng.randrange(spec.num_registers),
            rng.randrange(spec.num_registers),
        )
        pred = model.predict(state, ins)
        actual = cpu.step(state, ins)
        console.print(
            f"Instruction: {ins}\n"
            f"Predicted: {pred.registers}, Z={pred.zero}, C={pred.carry}\n"
            f"Actual:    {actual.registers}, Z={actual.zero}, C={actual.carry}"
        )
    else:
        instructions = _load_program(program)
        unknown = [ins.opcode for ins in instructions if ins.opcode not in oracle.valid_opcodes]
        if unknown:
            raise typer.BadParameter(
                "program contains opcode(s) not valid for this architecture: "
                + ", ".join(f"{opcode:02X}" for opcode in unknown)
            )
        state = CPUState(
            [rng.randint(0, spec.mask) for _ in range(spec.num_registers)],
            rng.randint(0, 1),
            rng.randint(0, 1),
            0,
        )
        pred = state.clone()
        actual = state.clone()
        for ins in instructions:
            pred = model.predict(pred, ins)
            actual = cpu.step(actual, ins)
        console.print(f"Initial:   {state.registers}, Z={state.zero}, C={state.carry}")
        console.print(f"Program:   {program}")
        console.print(f"Predicted: {pred.registers}, Z={pred.zero}, C={pred.carry}")
        console.print(f"Actual:    {actual.registers}, Z={actual.zero}, C={actual.carry}")

    exact = pred.registers == actual.registers and pred.zero == actual.zero and pred.carry == actual.carry
    console.print("[bold green]EXACT MATCH[/bold green]" if exact else "[bold red]MISMATCH[/bold red]")


@app.command()
def evaluate(
    checkpoint: str = typer.Option("", help="Optional learned Transformer checkpoint"),
    active: bool = typer.Option(False, "--active", help="Also evaluate ModelExplorer"),
):
    if checkpoint:
        from machinezero.evaluation.evaluate import evaluate_checkpoint

        results = evaluate_checkpoint(checkpoint, include_active=active)
        Path("results/eval.json").write_text(json.dumps(results, indent=2))
    else:
        from machinezero.evaluation.system_id_eval import evaluate as run_system_id

        results = run_system_id()
        Path("results/system_id_eval.json").write_text(json.dumps(results, indent=2))


@app.command()
def benchmark(n: int = typer.Option(20000)):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    subprocess.run(
        [sys.executable, "benchmarks/benchmark_simulator.py", "--n", str(n)],
        check=True,
        env=env,
    )


if __name__ == "__main__":
    app()
