from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from machinezero.aliencpu.generator import generate_architecture
from machinezero.aliencpu.oracle import HiddenOracle
from machinezero.aliencpu.simulator import AlienCPU
from machinezero.data.splits import make_splits
from machinezero.discovery.coverage import CoverageExplorer
from machinezero.discovery.random import RandomExplorer
from machinezero.evaluation.evaluate import sample_query
from machinezero.evaluation.metrics import compare_states
from machinezero.models.system_id import HypothesisPredictor


def evaluate(budgets=(0,1,2,5,10,20,30),queries_per_arch=40):
    splits=make_splits(202600,48,8,8); rows=[]
    for ename,explorer in [('random',RandomExplorer()),('coverage',CoverageExplorer())]:
        for budget in budgets:
            sums={'exact':0.,'register_accuracy':0.,'flag_accuracy':0.}; count=0
            for seed in splits.test:
                spec=generate_architecture(seed); cpu=AlienCPU(spec); oracle=HiddenOracle(cpu)
                ctx=explorer.discover(oracle,budget,seed=seed+budget).experiments if budget else []
                model=HypothesisPredictor(spec.word_bits,spec.num_registers).fit(ctx); rng=random.Random(seed*101+budget)
                for _ in range(queries_per_arch):
                    s,ins=sample_query(rng,spec); pred=model.predict(s,ins); actual=cpu.step(s,ins); m=compare_states(pred,actual)
                    for k in sums:sums[k]+=m[k]
                    count+=1
            row={'model':'system_id','explorer':ename,'budget':budget,'architectures':len(splits.test),'queries':count,**{k:v/count for k,v in sums.items()}}
            rows.append(row); print(row)
    return {'held_out_test_seeds':list(splits.test),'results':rows}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',default='results/system_id_eval.json'); a=p.parse_args(); r=evaluate(); Path(a.out).write_text(json.dumps(r,indent=2))
if __name__=='__main__': main()
