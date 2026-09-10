import pytest
import torch

from machinezero.aliencpu.batched_simulator import BatchedAlienCPU
from machinezero.aliencpu.generator import generate_architecture
from machinezero.aliencpu.instruction import Instruction
from machinezero.aliencpu.simulator import AlienCPU
from machinezero.aliencpu.state import CPUState


def test_batched_matches_scalar():
    spec = generate_architecture(123)
    scalar = AlienCPU(spec)
    batch = BatchedAlienCPU(spec, "cpu")
    regs = []
    ops = []
    aa = []
    bb = []
    expected = []
    for i, x in enumerate(spec.opcodes):
        rr = [(i * 7 + j * 3) & spec.mask for j in range(spec.num_registers)]
        ins = Instruction(x.opcode, i + 1, i + 2)
        regs.append(rr)
        ops.append(x.opcode)
        aa.append(ins.a)
        bb.append(ins.b)
        expected.append(scalar.step(CPUState(rr), ins))
    r, f, p = batch.step(torch.tensor(regs), torch.tensor(ops), torch.tensor(aa), torch.tensor(bb))
    for i, e in enumerate(expected):
        assert r[i].tolist() == e.registers
        assert f[i].tolist() == [e.zero, e.carry]
        assert p[i].item() == e.pc


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_cpu_parity():
    spec = generate_architecture(88)
    regs = torch.randint(0, spec.mask + 1, (64, spec.num_registers))
    op = torch.tensor([spec.opcodes[i % len(spec.opcodes)].opcode for i in range(64)])
    a = torch.arange(64)
    b = torch.arange(64) + 1
    c = BatchedAlienCPU(spec, "cpu").step(regs, op, a, b)
    g = BatchedAlienCPU(spec, "cuda").step(regs, op, a, b)
    assert torch.equal(c[0], g[0].cpu()) and torch.equal(c[1], g[1].cpu())
