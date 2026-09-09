# Evaluation

All primary metrics are computed by executing hidden simulators on architectures excluded from training.

Metrics:

- **Exact next-state accuracy:** all registers and flags match.
- **Register accuracy:** fraction of register values exactly correct.
- **Flag accuracy:** fraction of zero/carry flags correct.
- **Budget curves:** accuracy at 0, 1, 2, 5, 10, 20, and 30 experiments.
- **Explorer comparison:** random versus deterministic coverage probes; model-based exploration can be enabled separately.

Architecture-level splitting matters because transition-level random splitting would allow the learner to see the same opcode semantics during both training and test, defeating the central generalization question.
