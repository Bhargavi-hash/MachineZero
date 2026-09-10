from __future__ import annotations

from dataclasses import dataclass

from machinezero.aliencpu.instruction import Instruction
from machinezero.aliencpu.oracle import HiddenOracle
from machinezero.aliencpu.state import CPUState

Experiment = tuple[CPUState, Instruction, CPUState]


@dataclass
class DiscoveryResult:
    experiments: list[Experiment]


class Explorer:
    def discover(self, oracle: HiddenOracle, budget: int, seed: int = 0) -> DiscoveryResult:
        raise NotImplementedError
