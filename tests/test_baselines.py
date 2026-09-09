from machinezero.models.baselines import RandomStatePredictor,NoOpPredictor
from machinezero.aliencpu.state import CPUState

def test_baseline_shapes():
    assert len(RandomStatePredictor(8,4,1).predict().registers)==4
    s=CPUState([1,2,3,4]); assert NoOpPredictor(8).predict(s).registers==s.registers
