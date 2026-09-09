# System architecture

MachineZero separates **world generation**, **hidden execution**, **discovery**, **prediction**, and **verification**.

1. `generate_architecture(seed)` creates a complete immutable ISA specification.
2. `AlienCPU` executes that specification deterministically.
3. `HiddenOracle` wraps the simulator and exposes only word width, register count, valid opaque opcode bytes, and black-box `execute` calls.
4. An `Explorer` selects experiments under a hard budget.
5. A predictor consumes observed `(before, instruction, after)` transitions plus a new query.
6. Evaluation executes the same query on the hidden simulator and computes exact state metrics.

Architecture IDs are hashes of generated specifications for reproducibility. They are metadata only and are not model features.

## Modeling paths

The Transformer receives one structured token per observed transition and a query token. Its output is a bitwise state prediction. The enumerative system identifier instead maintains all candidate opcode semantics consistent with observations and predicts by majority over remaining hypotheses.

The latter is deliberately simple: it establishes that active black-box identification is possible before neural sophistication is added.
