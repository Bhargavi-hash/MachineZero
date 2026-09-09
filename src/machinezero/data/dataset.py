from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from machinezero.aliencpu.generator import generate_architecture
from machinezero.aliencpu.instruction import Instruction
from machinezero.aliencpu.simulator import AlienCPU
from machinezero.aliencpu.state import CPUState

from .encoding import encode_transition, semantic_target


class ProceduralTransitionDataset(Dataset):
    def __init__(
        self,
        architecture_seeds: list[int] | tuple[int, ...],
        samples_per_arch: int = 64,
        max_context: int = 5,
        seed: int = 0,
        focused_context_fraction: float = 0.65,
    ):
        self.architecture_seeds = tuple(architecture_seeds)
        self.samples_per_arch = samples_per_arch
        self.max_context = max_context
        self.seed = seed
        self.focused_context_fraction = focused_context_fraction

    def __len__(self) -> int:
        return len(self.architecture_seeds) * self.samples_per_arch

    @staticmethod
    def _state(rng, spec):
        return CPUState(
            [rng.randint(0, spec.mask) for _ in range(spec.num_registers)],
            rng.randint(0, 1),
            rng.randint(0, 1),
            0,
        )

    @staticmethod
    def _ins_for_spec(rng, spec, opcode_spec):
        if opcode_spec.operation == 'LOAD_IMMEDIATE':
            return Instruction(opcode_spec.opcode, rng.randrange(spec.num_registers), rng.randint(0, spec.mask))
        if opcode_spec.operation in {'JMP', 'JZ'}:
            return Instruction(opcode_spec.opcode, rng.randint(0, 15), 0)
        return Instruction(
            opcode_spec.opcode,
            rng.randrange(spec.num_registers),
            rng.randrange(spec.num_registers),
        )

    def __getitem__(self, idx: int):
        ai = idx // self.samples_per_arch
        arch_seed = self.architecture_seeds[ai]
        rng = random.Random((self.seed + 1) * 1_000_003 + idx * 7919)
        spec = generate_architecture(arch_seed)
        cpu = AlienCPU(spec)

        query_spec = rng.choice(spec.opcodes)
        query_before = self._state(rng, spec)
        query_ins = self._ins_for_spec(rng, spec, query_spec)

        k = rng.randint(0, self.max_context)
        tokens = []
        for i in range(k):
            # Since opcode identities are randomized independently per architecture,
            # useful meta-learning examples must sometimes demonstrate the queried
            # opcode. This is not a semantic leak: only observed transitions enter
            # the model, exactly as they do through the evaluation oracle.
            if i == 0 or rng.random() < self.focused_context_fraction:
                context_spec = query_spec
            else:
                context_spec = rng.choice(spec.opcodes)
            before = self._state(rng, spec)
            ins = self._ins_for_spec(rng, spec, context_spec)
            after = cpu.step(before, ins)
            tokens.append(
                encode_transition(
                    before, ins, after, spec.word_bits, spec.num_registers, query_opcode=query_ins.opcode
                )
            )

        tokens.append(
            encode_transition(
                query_before,
                query_ins,
                None,
                spec.word_bits,
                spec.num_registers,
                True,
                query_opcode=query_ins.opcode,
            )
        )
        pad = torch.zeros((self.max_context + 1, len(tokens[0])), dtype=torch.float32)
        padding_mask = torch.ones(self.max_context + 1, dtype=torch.bool)
        pad[: len(tokens)] = torch.stack(tokens)
        padding_mask[: len(tokens)] = False
        return {
            'tokens': pad,
            'padding_mask': padding_mask,
            'target': semantic_target(query_spec),
            'word_bits': spec.word_bits,
            'num_registers': spec.num_registers,
            'architecture_id': spec.architecture_id,
        }
