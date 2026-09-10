# Evaluation

All primary metrics are computed by executing hidden simulators on architectures excluded from training.

## Architecture-level isolation

The default split contains 48 train, 8 validation, and 8 test architectures. Entire generated architectures are assigned to one split. A transition-level random split would leak an opcode mapping into both training and test and therefore invalidate the central generalization question.

## Metrics

- **Exact next-state accuracy:** every output register and exposed flag matches.
- **Register accuracy:** fraction of registers whose integer value matches exactly.
- **Flag accuracy:** fraction of zero/carry values matching.
- **Operation accuracy:** learned operation family matches ground truth. Ground truth is used only for evaluation.
- **Semantic exact:** operation, operand direction, zero-update behavior, and carry-update behavior all match.
- **Budget curves:** 0, 1, 2, 5, 10, 20, and 30 black-box experiments.
- **Explorer comparison:** random, deterministic coverage, and model-informed active probes.

The learned evaluation also reports whether the queried opcode was actually observed in the context and separates exact accuracy for observed versus unobserved query opcodes. This is important because opaque opcode meaning is independently randomized for every architecture.

## Baselines

`machinezero.evaluation.baseline_eval` evaluates a random-state predictor and a no-op predictor. The neural budget-0 row is the no-context learned baseline. `machinezero.evaluation.system_id_eval` evaluates the explicit black-box hypothesis model.

## Executable verification

There is no LLM judge. A prediction is scored only by executing the hidden AlienCPU simulator and comparing concrete machine state.
