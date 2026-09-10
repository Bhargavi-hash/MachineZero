from __future__ import annotations

import math
import random
from collections import Counter

import torch

from machinezero.aliencpu.instruction import Instruction
from machinezero.aliencpu.state import CPUState
from machinezero.data.encoding import OPERATIONS, encode_transition

from .base import DiscoveryResult, Explorer


class ModelExplorer(Explorer):
    """Choose probes whose queried opcode has high model semantic uncertainty.

    This strategy never sees the hidden ISA. It scores candidate experiments from the
    current black-box context, combines predictive entropy with a small novelty bonus,
    executes only the selected candidate, and repeats until the budget is exhausted.
    """

    def __init__(self, model, device: str = "cpu", candidates: int = 0, mc_samples: int = 0):
        self.model = model
        self.device = device
        # Retained for checkpoint/backward compatibility with the first MVP API.
        self.candidates = candidates
        self.mc_samples = mc_samples

    def _candidate(self, oracle, opcode: int, probe_index: int, rng: random.Random):
        mask = (1 << oracle.word_bits) - 1
        patterns = [(3, 5), (mask, 1), (0, 0), (2, 1), (1 << (oracle.word_bits - 1), 1)]
        va, vb = patterns[probe_index % len(patterns)]
        a = probe_index % oracle.num_registers
        b = (probe_index + 1) % oracle.num_registers
        regs = [0] * oracle.num_registers
        regs[a] = va
        regs[b] = vb
        state = CPUState(regs, zero=probe_index % 2, carry=(probe_index // 2) % 2, pc=0)
        # Keep register-like operands for ALU hypotheses; later probes also expose
        # useful immediate/boundary values through b.
        operand_b = b if probe_index == 0 else vb & 0xFF
        return state, Instruction(opcode, a, operand_b)

    def _entropy(self, context, state, ins, oracle) -> float:
        tokens = [
            encode_transition(a, b, c, oracle.word_bits, oracle.num_registers, query_opcode=ins.opcode)
            for a, b, c in context
        ]
        tokens.append(
            encode_transition(
                state,
                ins,
                None,
                oracle.word_bits,
                oracle.num_registers,
                True,
                query_opcode=ins.opcode,
            )
        )
        x = torch.stack(tokens)[None].to(self.device)
        padding_mask = torch.zeros((1, len(tokens)), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            logits = self.model(x, padding_mask)[0, : len(OPERATIONS)]
            p = logits.softmax(-1)
            entropy = -(p * p.clamp_min(1e-9).log()).sum().item() / math.log(len(OPERATIONS))
        return entropy

    def discover(self, oracle, budget: int, seed: int = 0) -> DiscoveryResult:
        rng = random.Random(seed)
        context = []
        counts: Counter[int] = Counter()
        self.model.eval()

        for _ in range(budget):
            scored = []
            for opcode in oracle.valid_opcodes:
                probe_index = counts[opcode]
                state, ins = self._candidate(oracle, opcode, probe_index, rng)
                entropy = self._entropy(context, state, ins, oracle)
                novelty = 1.0 / (1.0 + counts[opcode])
                scored.append((entropy + 0.20 * novelty, opcode, state, ins))
            _, opcode, state, ins = max(scored, key=lambda row: (row[0], -row[1]))
            context.append((state, ins, oracle.execute(state, ins)))
            counts[opcode] += 1

        return DiscoveryResult(context)
