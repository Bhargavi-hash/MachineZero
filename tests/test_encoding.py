import torch

from machinezero.aliencpu.instruction import Instruction
from machinezero.aliencpu.state import CPUState
from machinezero.data.encoding import FEATURE_DIM, OPCODE_VOCAB, encode_transition


def test_encoding_has_no_seed_or_architecture_id_and_opcode_is_identity():
    before = CPUState([1, 2, 3, 4])
    after = CPUState([2, 2, 3, 4])
    a = encode_transition(before, Instruction(17, 0, 1), after, 8, 4)
    b = encode_transition(before, Instruction(18, 0, 1), after, 8, 4)
    assert a.shape == (FEATURE_DIM,)
    assert b.shape == (FEATURE_DIM,)
    assert not torch.equal(a, b)
    # Opcode one-hot occupies the 256-wide block after regs/flags/pc.
    opcode_start = 2 * 8 + 4 + 2
    assert a[opcode_start : opcode_start + OPCODE_VOCAB].sum().item() == 1.0
    assert a[opcode_start + 17].item() == 1.0
