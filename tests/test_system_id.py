from machinezero.aliencpu.generator import generate_architecture
from machinezero.aliencpu.oracle import HiddenOracle
from machinezero.aliencpu.simulator import AlienCPU
from machinezero.discovery.coverage import CoverageExplorer
from machinezero.models.system_id import HypothesisPredictor


def test_hypotheses_shrink_with_observations():
    spec = generate_architecture(11)
    oracle = HiddenOracle(AlienCPU(spec))
    ctx = CoverageExplorer().discover(oracle, 20).experiments
    model = HypothesisPredictor(spec.word_bits, spec.num_registers)
    op = spec.opcodes[0].opcode
    before = len(model.hypotheses(op))
    model.fit(ctx)
    after = len(model.hypotheses(op))
    assert after <= before
