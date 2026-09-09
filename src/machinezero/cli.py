from __future__ import annotations

import json
import random
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

app=typer.Typer(no_args_is_help=True,help='MachineZero: learning to understand computers never seen before.')
c=Console()

def _summary(spec,reveal=False):
    c.print(f'[bold]AlienCPU #{spec.architecture_id}[/bold]'); c.print(f'Registers: {spec.num_registers}\nWord width: {spec.word_bits} bits\nOpcodes: {len(spec.opcodes)}\nArchitecture semantics: {"REVEALED" if reveal else "HIDDEN"}')
    if reveal:
        t=Table('Opcode','Operation','Dst first','Z','C')
        for x in spec.opcodes:t.add_row(f'{x.opcode:02X}',x.operation,str(x.dst_first),str(x.updates_zero),str(x.updates_carry))
        c.print(t)

@app.command()
def info(): c.print('[bold]MachineZero[/bold]\nLearning to understand computers never seen before.\nCore environment: AlienCPU')
@app.command()
def generate(seed:int=typer.Option(42)):
    _summary(generate_architecture(seed),False)
@app.command()
def inspect(seed:int=typer.Option(42),reveal:bool=typer.Option(False,'--reveal')):
    _summary(generate_architecture(seed),reveal)
@app.command()
def discover(seed:int=typer.Option(42),budget:int=typer.Option(20),explorer:str=typer.Option('coverage')):
    spec=generate_architecture(seed); oracle=HiddenOracle(AlienCPU(spec)); ex=CoverageExplorer() if explorer=='coverage' else RandomExplorer(); ctx=ex.discover(oracle,budget,seed).experiments; model=HypothesisPredictor(spec.word_bits,spec.num_registers).fit(ctx); _summary(spec,False)
    c.print(f'\n[bold]Experiments: {budget} ({explorer})[/bold]')
    for i,(before,ins,after) in enumerate(ctx[:min(8,len(ctx))],1): c.print(f'{i:02d}. {ins}  {before.registers} -> {after.registers}')
    t=Table('Opcode','Likely operation','Candidates','Confidence')
    for op in oracle.valid_opcodes:
        h=model.summarize(op); t.add_row(f'{op:02X}',h['likely_operation'],str(h['candidate_count']),f"{h['confidence']:.2f}")
    c.print('\n[bold]Learned hypotheses[/bold]'); c.print(t)
@app.command()
def predict(seed:int=typer.Option(42),budget:int=typer.Option(20)):
    spec=generate_architecture(seed); cpu=AlienCPU(spec); oracle=HiddenOracle(cpu); ctx=CoverageExplorer().discover(oracle,budget,seed).experiments; model=HypothesisPredictor(spec.word_bits,spec.num_registers).fit(ctx); rng=random.Random(seed+999); sp=rng.choice(spec.opcodes); s=CPUState([rng.randint(0,spec.mask) for _ in range(spec.num_registers)],rng.randint(0,1),rng.randint(0,1)); ins=Instruction(sp.opcode,rng.randrange(spec.num_registers),rng.randrange(spec.num_registers)); pred=model.predict(s,ins); actual=cpu.step(s,ins); c.print(f'Instruction: {ins}\nPredicted: {pred.registers}, Z={pred.zero}, C={pred.carry}\nActual:    {actual.registers}, Z={actual.zero}, C={actual.carry}'); c.print('[bold green]EXACT MATCH[/bold green]' if pred.registers==actual.registers and pred.zero==actual.zero and pred.carry==actual.carry else '[bold red]MISMATCH[/bold red]')
@app.command()
def evaluate(checkpoint:str=typer.Option('',help='Optional Transformer checkpoint')):
    if checkpoint:
        from machinezero.evaluation.evaluate import evaluate_checkpoint
        r=evaluate_checkpoint(checkpoint); Path('results/eval.json').write_text(json.dumps(r,indent=2))
    else:
        from machinezero.evaluation.system_id_eval import evaluate as run
        r=run(); Path('results/system_id_eval.json').write_text(json.dumps(r,indent=2))
@app.command()
def benchmark(n:int=typer.Option(20000)):
    import os
    import subprocess
    import sys
    env=dict(os.environ); env['PYTHONPATH']=str(Path(__file__).resolve().parents[2])
    subprocess.run([sys.executable,'benchmarks/benchmark_simulator.py','--n',str(n)],check=True,env=env)

if __name__ == '__main__':
    app()
