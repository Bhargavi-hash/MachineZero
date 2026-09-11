from __future__ import annotations

import random
from pathlib import Path

import torch
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from machinezero.aliencpu.generator import generate_architecture
from machinezero.aliencpu.instruction import Instruction
from machinezero.aliencpu.oracle import HiddenOracle
from machinezero.aliencpu.simulator import AlienCPU
from machinezero.aliencpu.state import CPUState
from machinezero.discovery.coverage import CoverageExplorer
from machinezero.models.checkpoint import load_checkpoint
from machinezero.models.prediction import predict_state

app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console()


def _checkpoint_path(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return path
        raise typer.BadParameter(f"Checkpoint not found: {path}")

    candidates = [
        Path.home() / ".cache" / "machinezero" / "models" / "small" / "model.pt",
        Path("checkpoints/model.pt"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise typer.BadParameter(
        "No MachineZero-Small checkpoint found. Run `machinezero pull small` "
        "or pass --checkpoint PATH."
    )


def _state_text(state: CPUState) -> str:
    return f"{state.registers}  Z={state.zero} C={state.carry} PC={state.pc}"


def _exact(a: CPUState, b: CPUState) -> bool:
    return (
        a.registers == b.registers
        and a.zero == b.zero
        and a.carry == b.carry
        and a.pc == b.pc
    )


@app.callback(invoke_without_command=True)
def main(
    seed: int = typer.Option(2026, help="Held-out AlienCPU seed."),
    budget: int = typer.Option(20, help="Black-box experiment budget."),
    checkpoint: str | None = typer.Option(None, help="Path to MachineZero-Small model.pt."),
) -> None:
    console.print(
        Panel.fit(
            "[bold]MACHINEZERO[/bold]\n[dim]Learning a computer never seen before[/dim]",
            border_style="cyan",
        )
    )

    spec = generate_architecture(seed)
    cpu = AlienCPU(spec)
    oracle = HiddenOracle(cpu)

    info = Table(show_header=False, box=None, pad_edge=False)
    info.add_row("Architecture", f"#{spec.architecture_id}")
    info.add_row("Word size", f"{spec.word_bits} bits")
    info.add_row("Registers", str(spec.num_registers))
    info.add_row("Unknown opcodes", str(len(spec.opcodes)))
    info.add_row("Hidden ISA", "  ".join(f"0x{op:02X} → ???" for op in oracle.valid_opcodes))
    console.print("\n[bold cyan]NEW ALIEN CPU[/bold cyan]")
    console.print(info)
    console.print("[dim]MachineZero receives no opcode definitions. Only black-box execution access.[/dim]")

    result = CoverageExplorer().discover(oracle, budget, seed)
    context = result.experiments

    console.print(f"\n[bold cyan]EXPERIMENTING[/bold cyan]  [dim]budget={budget}[/dim]")
    shown = min(6, len(context))
    for idx, (before, ins, after) in enumerate(context[:shown], 1):
        console.print(
            f"[{idx:02d}/{budget:02d}] "
            f"0x{ins.opcode:02X} ({ins.a}, {ins.b})   "
            f"{before.registers}  →  {after.registers}"
        )
    if len(context) > shown:
        console.print(f"[dim]      … {len(context) - shown} more black-box experiments[/dim]")
    console.print(f"[green]✓[/green] {len(context)} experiments complete")

    ckpt = _checkpoint_path(checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, metadata = load_checkpoint(str(ckpt), device=device)
    parameter_count = metadata.get("parameter_count", sum(p.numel() for p in model.parameters()))

    model_table = Table(show_header=False, box=None, pad_edge=False)
    model_table.add_row("Open model", "MachineZero-Small")
    model_table.add_row("Parameters", f"{parameter_count:,}")
    model_table.add_row("Checkpoint", str(ckpt))
    model_table.add_row("Device", device.upper())
    console.print("\n[bold cyan]OPEN MODEL[/bold cyan]")
    console.print(model_table)

    rng = random.Random(seed + 999)
    opcode = rng.choice(list(oracle.valid_opcodes))
    before = CPUState(
        [rng.randint(0, spec.mask) for _ in range(spec.num_registers)],
        rng.randint(0, 1),
        rng.randint(0, 1),
        0,
    )
    instruction = Instruction(
        opcode,
        rng.randrange(spec.num_registers),
        rng.randrange(spec.num_registers),
    )

    predicted = predict_state(
        model,
        context,
        before,
        instruction,
        spec.word_bits,
        spec.num_registers,
        device=device,
    )
    actual = cpu.step(before, instruction)

    test = Table(show_header=False, box=None, pad_edge=False)
    test.add_row("Instruction", f"0x{instruction.opcode:02X}  {instruction.a}  {instruction.b}")
    test.add_row("Initial", _state_text(before))
    test.add_row("Predicted", _state_text(predicted))
    test.add_row("Actual", _state_text(actual))
    console.print("\n[bold cyan]UNSEEN TEST[/bold cyan]")
    console.print(test)

    if _exact(predicted, actual):
        console.print(Panel.fit("[bold green]✓ EXACT MATCH[/bold green]", border_style="green"))
    else:
        console.print(Panel.fit("[bold yellow]PREDICTION ≠ EXACT STATE[/bold yellow]", border_style="yellow"))
        console.print(
            "[dim]This is a learned-model prediction, so individual held-out queries can fail. "
            "Use the aggregate held-out evaluation for the reported accuracy.[/dim]"
        )


if __name__ == "__main__":
    app()
