from __future__ import annotations
import argparse,json,random
from pathlib import Path
import yaml, torch
from torch.utils.data import DataLoader
from machinezero.data.splits import make_splits,validate_disjoint
from machinezero.data.dataset import ProceduralTransitionDataset
from machinezero.models.transformer import MachineZeroTransformer
from machinezero.models.checkpoint import save_checkpoint
from .losses import state_bit_loss

def main():
 p=argparse.ArgumentParser(); p.add_argument('--data-config',default='configs/data/small.yaml'); p.add_argument('--model-config',default='configs/model/small.yaml'); p.add_argument('--train-config',default='configs/train/default.yaml'); p.add_argument('--out',default='checkpoints/model.pt'); a=p.parse_args()
 dc=yaml.safe_load(Path(a.data_config).read_text()); mc=yaml.safe_load(Path(a.model_config).read_text()); tc=yaml.safe_load(Path(a.train_config).read_text()); random.seed(tc['seed']); torch.manual_seed(tc['seed'])
 device=torch.device('cuda' if torch.cuda.is_available() and not tc.get('cpu_only',False) else 'cpu'); splits=make_splits(dc['base_seed'],dc['train_architectures'],dc['validation_architectures'],dc['test_architectures']); validate_disjoint(splits)
 ds=ProceduralTransitionDataset(splits.train,dc['samples_per_architecture'],dc['max_context'],tc['seed']); dl=DataLoader(ds,batch_size=tc['batch_size'],shuffle=True,num_workers=0)
 model=MachineZeroTransformer(**mc).to(device); opt=torch.optim.AdamW(model.parameters(),lr=tc['learning_rate'])
 print(f'device={device} parameters={model.parameter_count():,} samples={len(ds)}')
 model.train(); step=0
 for epoch in range(tc['epochs']):
  total=0.0
  for batch in dl:
   opt.zero_grad(); y=model(batch['tokens'].to(device),batch['padding_mask'].to(device)); loss=state_bit_loss(y,batch['target'].to(device)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); total+=loss.item(); step+=1
  print(f'epoch={epoch+1} loss={total/len(dl):.6f}')
 Path(a.out).parent.mkdir(parents=True,exist_ok=True); meta={'model_config':mc,'data_config':dc,'train_config':tc,'device':str(device),'parameter_count':model.parameter_count()}; save_checkpoint(a.out,model,meta); Path('results/train.json').write_text(json.dumps(meta,indent=2)); print(f'saved={a.out}')
if __name__=='__main__': main()
