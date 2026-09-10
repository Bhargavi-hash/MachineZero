# MachineZero Small v0.1

MachineZero studies whether a learned system can understand a previously
unseen computational architecture by actively experimenting with it.

> It isn't learning an ISA. It's learning how to learn one.

## Model

- Architecture: MachineZero structured Transformer
- Parameters: 1,182,064
- Benchmark: AlienCPU
- Evaluation split: architecture-level held-out CPUs
- Training architectures: 48
- Test architectures: 8

## Inputs

The model receives observed transitions from an unseen AlienCPU:

state + instruction -> next_state

and a query transition to predict.

No architecture seed, ISA specification, or hidden opcode semantics are
provided as model inputs.

## Measured results

CoverageExplorer exact next-state accuracy:

| Experiments | Exact accuracy |
|---:|---:|
| 0 | 15.9% |
| 5 | 30.6% |
| 10 | 59.4% |
| 20 | 66.6% |

The transparent system-identification baseline reaches 90.0% exact accuracy
at budget 20 and 94.1% at budget 30.

These numbers are measured on held-out AlienCPU architectures.

## Code

The full MachineZero source code, AlienCPU environment, training pipeline,
evaluation harness, and demo are available in the MachineZero GitHub repository.

## Limitations

This is a research prototype, not a production CPU emulator.

The learned model currently underperforms the explicit system-identification
baseline.

Program synthesis is not part of v0.1.

## License

Apache-2.0

## Author

Bhargavi Kurukunda
