from machinezero.aliencpu.generator import generate_architecture
from machinezero.aliencpu.instruction import Instruction
from machinezero.aliencpu.simulator import AlienCPU
from machinezero.aliencpu.state import CPUState


def find(spec,name): return next(x for x in spec.opcodes if x.operation==name)
def test_wrap_and_add():
 spec=generate_architecture(7); cpu=AlienCPU(spec); add=find(spec,'ADD')
 s=CPUState([spec.mask,1]+[0]*(spec.num_registers-2)); a,b=(0,1) if add.dst_first else (1,0)
 out=cpu.step(s,Instruction(add.opcode,a,b)); assert 0 in out.registers[:2]
def test_load_imm():
 spec=generate_architecture(9); cpu=AlienCPU(spec); li=find(spec,'LOAD_IMMEDIATE')
 out=cpu.step(CPUState([0]*spec.num_registers),Instruction(li.opcode,1,spec.mask+4)); assert out.registers[1]==3
