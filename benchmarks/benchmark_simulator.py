from __future__ import annotations
import argparse,time,json
from pathlib import Path
import torch
from machinezero.aliencpu.generator import generate_architecture
from machinezero.aliencpu.simulator import AlienCPU
from machinezero.aliencpu.batched_simulator import BatchedAlienCPU
from machinezero.aliencpu.state import CPUState
from machinezero.aliencpu.instruction import Instruction

def bench_scalar(spec,n:int):
    cpu=AlienCPU(spec); states=[CPUState([(i+j)&spec.mask for j in range(spec.num_registers)]) for i in range(n)]; ins=[Instruction(spec.opcodes[i%len(spec.opcodes)].opcode,i%spec.num_registers,(i+1)%spec.num_registers) for i in range(n)]
    t=time.perf_counter()
    for s,x in zip(states,ins): cpu.step(s,x)
    return n/(time.perf_counter()-t)

def bench_batch(spec,n:int,device:str):
    sim=BatchedAlienCPU(spec,device); regs=torch.randint(0,spec.mask+1,(n,spec.num_registers)); ops=torch.tensor([spec.opcodes[i%len(spec.opcodes)].opcode for i in range(n)]); a=torch.arange(n)%spec.num_registers; b=(torch.arange(n)+1)%spec.num_registers
    if device=='cuda': regs,ops,a,b=regs.cuda(),ops.cuda(),a.cuda(),b.cuda(); torch.cuda.synchronize()
    for _ in range(2): sim.step(regs,ops,a,b)
    if device=='cuda': torch.cuda.synchronize()
    t=time.perf_counter()
    for _ in range(5): sim.step(regs,ops,a,b)
    if device=='cuda': torch.cuda.synchronize()
    return (n*5)/(time.perf_counter()-t)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--n',type=int,default=20000); p.add_argument('--out',default='results/benchmark.json'); a=p.parse_args(); spec=generate_architecture(42)
    r={'n':a.n,'scalar_python_tps':bench_scalar(spec,a.n),'batched_torch_cpu_tps':bench_batch(spec,a.n,'cpu'),'cuda_available':torch.cuda.is_available()}
    r['batched_torch_cuda_tps']=bench_batch(spec,a.n,'cuda') if torch.cuda.is_available() else None
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
if __name__=='__main__': main()
