# AlienCPU

AlienCPU generates synthetic fixed-width computational architectures from deterministic seeds.

Current randomized dimensions include 8/12/16-bit words, 4-8 registers, 8-13 selected instructions, opaque opcode byte assignments, operand ordering, zero/carry updates, and branches. Core operation families include MOV, arithmetic, Boolean operations, shifts, immediate loads, compare, and jumps.

`ArchitectureSpec` is serializable ground truth. It is used for tests and `inspect --reveal`, but is never handed to discovery agents during evaluation.

## State

A state contains registers, zero/carry flags, and a program counter. Memory is intentionally omitted from the MVP.

## Train/test isolation

`make_splits` assigns entire architecture seeds to train, validation, and test partitions. No architecture can contribute transitions to more than one split.
