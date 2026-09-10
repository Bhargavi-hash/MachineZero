from machinezero.aliencpu.generator import generate_architecture
from machinezero.aliencpu.instruction import Instruction
from machinezero.aliencpu.simulator import AlienCPU
from machinezero.aliencpu.state import CPUState
from machinezero.models.system_id import HypothesisPredictor


def test_system_identifier_predicts_pc_once():
    spec = generate_architecture(2026)
    cpu = AlienCPU(spec)
    op = spec.opcodes[0]
    state = CPUState([1] * spec.num_registers, 0, 0, 0)
    ins = Instruction(op.opcode, 0, 1)
    after = cpu.step(state, ins)
    model = HypothesisPredictor(spec.word_bits, spec.num_registers).fit([(state, ins, after)])
    pred = model.predict(state, ins)
    assert pred.pc == after.pc
