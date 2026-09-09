from machinezero.aliencpu.generator import generate_architecture
from machinezero.aliencpu.instruction import Instruction
from machinezero.aliencpu.simulator import AlienCPU
from machinezero.aliencpu.state import CPUState
from machinezero.models.prediction import execute_hypothesis


def test_inferred_executor_matches_ground_truth_for_known_semantics():
    spec = generate_architecture(2026)
    cpu = AlienCPU(spec)
    state = CPUState([i + 2 for i in range(spec.num_registers)], zero=1, carry=0, pc=0)
    for op in spec.opcodes:
        ins = Instruction(op.opcode, 0, 1)
        expected = cpu.step(state, ins)
        actual = execute_hypothesis(
            state,
            ins,
            spec.word_bits,
            spec.num_registers,
            op.operation,
            op.dst_first,
            op.updates_zero,
            op.updates_carry,
        )
        assert actual == expected
