# Models

## Transformer baseline

The neural baseline is a compact PyTorch Transformer encoder. Each context token contains padded normalized registers before/after an experiment, flags, PC, opaque opcode byte, operands, word width, and register count. The query token contains no after-state.

The architecture seed, architecture ID, and hidden ISA semantics are not encoded.

The current output objective predicts 16 bits for each of eight padded register slots plus two flag bits using binary cross entropy. This first objective trains, but the captured small run does not generalize well enough for the main demo.

## System-identification baseline

`HypothesisPredictor` enumerates candidate operation/operand/flag-update hypotheses and removes any candidate inconsistent with observed transitions. It never reads the true opcode table.

This baseline is useful scientifically because it provides an interpretable ceiling for the current AlienCPU complexity and directly measures the value of experiment selection.

## Active exploration

`ModelExplorer` samples candidate experiments and uses MC-dropout disagreement from the Transformer as a simple uncertainty score. It is implemented as an experimental baseline; no superiority claim is made yet.
