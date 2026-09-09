from __future__ import annotations
import torch
from .architecture import ArchitectureSpec

OP_IDS={'MOV':0,'ADD':1,'SUB':2,'XOR':3,'AND':4,'OR':5,'NOT':6,'SHL':7,'SHR':8,'LOAD_IMMEDIATE':9,'COMPARE':10,'JMP':11,'JZ':12}

class BatchedAlienCPU:
    '''Vectorized transition simulator for one AlienCPU architecture.'''
    def __init__(self,spec:ArchitectureSpec,device:str|torch.device|None=None):
        self.spec=spec; self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        lut_size=256
        self.op_id=torch.full((lut_size,),-1,dtype=torch.long,device=self.device)
        self.dst_first=torch.ones(lut_size,dtype=torch.bool,device=self.device)
        self.upz=torch.zeros(lut_size,dtype=torch.bool,device=self.device)
        self.upc=torch.zeros(lut_size,dtype=torch.bool,device=self.device)
        for x in spec.opcodes:
            self.op_id[x.opcode]=OP_IDS[x.operation]; self.dst_first[x.opcode]=x.dst_first; self.upz[x.opcode]=x.updates_zero; self.upc[x.opcode]=x.updates_carry

    def step(self,registers:torch.Tensor,opcodes:torch.Tensor,a:torch.Tensor,b:torch.Tensor,flags:torch.Tensor|None=None,pc:torch.Tensor|None=None):
        r=registers.to(self.device).long().clone(); oc=opcodes.to(self.device).long(); a=a.to(self.device).long(); b=b.to(self.device).long()
        n=r.shape[0]; flags=torch.zeros((n,2),dtype=torch.long,device=self.device) if flags is None else flags.to(self.device).long().clone()
        pc=torch.zeros(n,dtype=torch.long,device=self.device) if pc is None else pc.to(self.device).long().clone()
        oid=self.op_id[oc]; df=self.dst_first[oc]; ai=a%self.spec.num_registers; bi=b%self.spec.num_registers
        dst=torch.where(df,ai,bi); src=torch.where(df,bi,ai); rows=torch.arange(n,device=self.device)
        old=r[rows,dst]; rhs=r[rows,src]; mask=self.spec.mask
        result=old.clone(); writes=torch.zeros(n,dtype=torch.bool,device=self.device); raw=old.clone(); carry=flags[:,1].clone()
        def setop(k,val):
            nonlocal result,writes
            m=oid==k; result=torch.where(m,val,result); writes|=m; return m
        setop(0,rhs); m=setop(1,(old+rhs)&mask); raw=old+rhs; carry=torch.where(m,(raw>mask).long(),carry)
        m=setop(2,(old-rhs)&mask); carry=torch.where(m,(old>=rhs).long(),carry)
        setop(3,old^rhs); setop(4,old&rhs); setop(5,old|rhs); setop(6,(~old)&mask)
        m=setop(7,(old<<1)&mask); carry=torch.where(m,((old>>(self.spec.word_bits-1))&1),carry)
        m=setop(8,(old>>1)&mask); carry=torch.where(m,old&1,carry)
        m=oid==9; imm=b&mask; result=torch.where(m,imm,result); dst=torch.where(m,ai,dst); writes|=m
        cmp=oid==10; result=torch.where(cmp,(old-rhs)&mask,result); carry=torch.where(cmp,(old>=rhs).long(),carry)
        r[rows,dst]=torch.where(writes,result,r[rows,dst])
        upz=self.upz[oc] & (writes|cmp); flags[:,0]=torch.where(upz,(result==0).long(),flags[:,0])
        flags[:,1]=torch.where(self.upc[oc],carry,flags[:,1])
        jmp=oid==11; jz=(oid==12)&(flags[:,0]==1); taken=jmp|jz
        pc=torch.where(taken,a&mask,(pc+1)&mask)
        return r,flags,pc
