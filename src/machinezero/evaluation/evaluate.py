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
from machinezero.models.prediction import predict_state

from .metrics import compare_states


def sample_query(rng,spec):
 s=CPUState([rng.randint(0,spec.mask) for _ in range(spec.num_registers)],rng.randint(0,1),rng.randint(0,1),0); sp=rng.choice(spec.opcodes)
 if sp.operation=='LOAD_IMMEDIATE': ins=Instruction(sp.opcode,rng.randrange(spec.num_registers),rng.randint(0,spec.mask))
 elif sp.operation in {'JMP','JZ'}: ins=Instruction(sp.opcode,rng.randint(0,15),0)
 else: ins=Instruction(sp.opcode,rng.randrange(spec.num_registers),rng.randrange(spec.num_registers))
 return s,ins

def evaluate_checkpoint(path:str,budgets=(0,1,2,5,10,20),queries_per_arch=12,include_active=False):
 device='cuda' if torch.cuda.is_available() else 'cpu'; model,meta=load_checkpoint(path,device); dc=meta['data_config']; splits=make_splits(dc['base_seed'],dc['train_architectures'],dc['validation_architectures'],dc['test_architectures'])
 explorers={'random':RandomExplorer(),'coverage':CoverageExplorer()};
 if include_active: explorers['model']=ModelExplorer(model,device=device,candidates=8,mc_samples=3)
 rows=[]
 for name,explorer in explorers.items():
  for budget in budgets:
   sums={'exact':0.0,'register_accuracy':0.0,'flag_accuracy':0.0}; count=0
   for arch_seed in splits.test:
    spec=generate_architecture(arch_seed); cpu=AlienCPU(spec); oracle=HiddenOracle(cpu); ctx=explorer.discover(oracle,budget,seed=arch_seed+budget).experiments if budget else []; rng=random.Random(arch_seed*31+budget)
    for _ in range(queries_per_arch):
      s,ins=sample_query(rng,spec); actual=cpu.step(s,ins); pred=predict_state(model,ctx,s,ins,spec.word_bits,spec.num_registers,device); m=compare_states(pred,actual)
      for k in sums:sums[k]+=m[k]
      count+=1
   row={'explorer':name,'budget':budget,'architectures':len(splits.test),'queries':count,**{k:v/count for k,v in sums.items()}}; rows.append(row); print(row)
 return {'checkpoint':path,'device':device,'held_out_test_seeds':list(splits.test),'results':rows}

def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--out',default='results/eval.json'); p.add_argument('--active',action='store_true'); a=p.parse_args(); r=evaluate_checkpoint(a.checkpoint,include_active=a.active); Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(r,indent=2))
if __name__=='__main__':main()
