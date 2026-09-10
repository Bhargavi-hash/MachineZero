from __future__ import annotations

import json
import random
from pathlib import Path

import typer
from rich.table import Table

from machinezero.aliencpu.generator import generate_architecture
from machinezero.aliencpu.instruction import Instruction
from machinezero.aliencpu.oracle import HiddenOracle
from machinezero.aliencpu.simulator import AlienCPU
from machinezero.aliencpu.state import CPUState
from machinezero.cli import app, console
from machinezero.discovery.coverage import CoverageExplorer
from machinezero.models.registry import OFFICIAL_MODELS, model_checkpoint, pull_model, resolve_checkpoint

MODEL_ARGUMENT = typer.Argument("small", help="Official model alias or Hugging Face owner/repo")
PROGRAM_OPTION = typer.Option(None, "--program", exists=True, dir_okay=False, help="Optional .mz program")


@app.command("models")
def list_models():
    """List official pretrained models and local download status."""
    table = Table("Model", "Hugging Face repo", "Status", "Checkpoint")
    for name, spec in OFFICIAL_MODELS.items():
        checkpoint = model_checkpoint(name)
        table.add_row(
            name,
            spec.repo_id,
            "downloaded" if checkpoint.is_file() else "not downloaded",
            str(checkpoint),
        )
    console.print(table)


@app.command()
def pull(
    model: str = MODEL_ARGUMENT,
    force: bool = typer.Option(False, "--force", help="Redownload files even if cached"),
):
    """Download a pretrained MachineZero model from Hugging Face Hub."""
    try:
        spec, checkpoint = pull_model(model, force=force)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"[bold green]Ready[/bold green] {spec.repo_id}")
    console.print(f"Checkpoint: {checkpoint}")


@app.command("model-path")
def model_path(model: str = MODEL_ARGUMENT):
    """Print the local checkpoint path for a downloaded model."""
    try:
        checkpoint = resolve_checkpoint(model=model)
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(str(checkpoint))


@app.command("evaluate-model")
def evaluate_model(
    model: str = MODEL_ARGUMENT,
    active: bool = typer.Option(False, "--active", help="Also evaluate ModelExplorer"),
):
    """Evaluate a downloaded pretrained model."""
    from machinezero.evaluation.evaluate import evaluate_checkpoint

    try:
        checkpoint = resolve_checkpoint(model=model)
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    results = evaluate_checkpoint(str(checkpoint), include_active=active)
    Path("results").mkdir(exist_ok=True)
    Path("results/eval.json").write_text(json.dumps(results, indent=2))


def _parse_program(path: Path) -> list[Instruction]:
    instructions: list[Instruction] = []
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3:
            raise typer.BadParameter(f"{path}:{line_number}: expected OPCODE A B")
        try:
            opcode, a, b = (int(part, 16) for part in parts)
        except ValueError as exc:
            raise typer.BadParameter(f"{path}:{line_number}: values must be hexadecimal") from exc
        instructions.append(Instruction(opcode, a, b))
    if not instructions:
        raise typer.BadParameter(f"{path}: program contains no instructions")
    return instructions


@app.command("run")
def run_model(
    model: str = MODEL_ARGUMENT,
    program: Path | None = PROGRAM_OPTION,
    seed: int = typer.Option(2026),
    budget: int = typer.Option(20),
):
    """Run a downloaded learned model on an unseen AlienCPU query or .mz program."""
    import torch

    from machinezero.models.checkpoint import load_checkpoint
    from machinezero.models.prediction import predict_state

    try:
        checkpoint = resolve_checkpoint(model=model)
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    spec = generate_architecture(seed)
    cpu = AlienCPU(spec)
    context = CoverageExplorer().discover(HiddenOracle(cpu), budget, seed).experiments
    device = "cuda" if torch.cuda.is_available() else "cpu"
    learned_model, _ = load_checkpoint(str(checkpoint), device)
    learned_model.to(device).eval()

    rng = random.Random(seed + 999)
    state = CPUState(
        [rng.randint(0, spec.mask) for _ in range(spec.num_registers)],
        rng.randint(0, 1),
        rng.randint(0, 1),
    )

    def learned_predict(current: CPUState, instruction: Instruction) -> CPUState:
        return predict_state(
            learned_model,
            context,
            current,
            instruction,
            spec.word_bits,
            spec.num_registers,
            device,
        )

    if program is None:
        opcode = rng.choice(spec.opcodes).opcode
        instruction = Instruction(opcode, rng.randrange(spec.num_registers), rng.randrange(spec.num_registers))
        predicted = learned_predict(state, instruction)
        actual = cpu.step(state, instruction)
        console.print(f"Model:      {model}")
        console.print(f"Instruction: {instruction}")
    else:
        predicted = state.clone()
        actual = state.clone()
        for instruction in _parse_program(program):
            predicted = learned_predict(predicted, instruction)
            actual = cpu.step(actual, instruction)
        console.print(f"Model:     {model}")
        console.print(f"Program:   {program}")

    console.print(f"Predicted: {predicted.registers}, Z={predicted.zero}, C={predicted.carry}")
    console.print(f"Actual:    {actual.registers}, Z={actual.zero}, C={actual.carry}")
    exact = (
        predicted.registers == actual.registers and predicted.zero == actual.zero and predicted.carry == actual.carry
    )
    console.print("[bold green]EXACT MATCH[/bold green]" if exact else "[bold red]MISMATCH[/bold red]")
