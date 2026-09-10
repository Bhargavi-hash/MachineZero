# Models

MachineZero keeps the neural learner small and architecture-independent. The model never receives an architecture seed, architecture ID, or hidden opcode table.

## Structured semantic Transformer

The current learned model is a **1,182,064-parameter Transformer encoder**. Instead of reconstructing every output-state bit directly, it predicts a latent hypothesis for the queried opcode:

- operation family (`MOV`, `ADD`, `SUB`, `XOR`, ...)
- operand direction
- whether the zero flag is updated
- whether the carry flag is updated

The inferred hypothesis is then executed by a generic semantics executor to obtain the predicted next state. The hidden AlienCPU simulator is used only afterward for evaluation.

Each observed transition token contains padded registers before/after, flags, PC, the opaque opcode identity, operands, word width, register count, and a query-opcode match marker. Opcode bytes are one-hot identities rather than scalar magnitudes because `0x10` is not semantically closer to `0x11` than to `0xF0`.

### Observable candidate-consistency features

Raw normalized values made the first neural baseline difficult to train: the network had to rediscover modular arithmetic before it could identify an operation. The current representation therefore adds a compact set of **observable hypothesis-consistency features**. For each generic operation and operand direction, the encoder asks whether that hypothesis could have produced the observed register/PC transition.

These features do **not** use the generated ISA specification. They are computed solely from:

```text
(before state, opaque instruction, observed after state, public machine dimensions)
```

This is an explicit inductive bias toward system identification. It makes ambiguity visible to the learner while preserving the central hidden-semantics constraint.

The Transformer explicitly pools observations whose opaque opcode matches the query opcode and fuses that evidence with the query representation.

## Training objective

Training uses architecture-level splits. The semantic operation is trained with categorical cross entropy and the three binary attributes with binary cross entropy.

Training contexts are query-focused some of the time. This is necessary because opcode mappings are independently randomized per architecture: if an opcode has never been observed on a new CPU, its local meaning cannot be inferred from its byte value alone.

The small reproducible run uses 48 training architectures, 8 validation architectures, and 8 completely held-out test architectures.

## System-identification baseline

`HypothesisPredictor` enumerates generic operation/operand/flag hypotheses and removes candidates inconsistent with observed transitions. It never reads the true opcode table.

This baseline is intentionally transparent. It establishes that AlienCPU is identifiable under a small interaction budget and gives a useful reference point for the learned model.

## Active exploration

Three explorers are implemented:

- `RandomExplorer`: random states/instructions.
- `CoverageExplorer`: deterministic probes chosen to expose operation differences, wrapping, boundary behavior, and flags.
- `ModelExplorer`: scores opaque opcodes using the learned model's operation entropy plus a small novelty bonus, executes only the highest-scoring probe, observes the transition, and repeats.

`ModelExplorer` is model-informed but intentionally simple; it is not presented as an optimal active-learning algorithm.
