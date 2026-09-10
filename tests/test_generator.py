from machinezero.aliencpu.generator import generate_architecture


def test_deterministic():
    assert generate_architecture(42) == generate_architecture(42)


def test_different():
    a, b = generate_architecture(1), generate_architecture(2)
    assert a.architecture_id != b.architecture_id and a.opcodes != b.opcodes
