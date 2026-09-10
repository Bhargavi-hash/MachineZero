from machinezero.aliencpu.generator import generate_architecture
from machinezero.aliencpu.oracle import HiddenOracle
from machinezero.aliencpu.simulator import AlienCPU
from machinezero.discovery.coverage import CoverageExplorer
from machinezero.discovery.random import RandomExplorer


def test_budget_respected():
    o = HiddenOracle(AlienCPU(generate_architecture(3)))
    assert len(RandomExplorer().discover(o, 7).experiments) == 7
    assert len(CoverageExplorer().discover(o, 9).experiments) == 9
