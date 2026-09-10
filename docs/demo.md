# Demo

The deterministic contest demo uses a completely generated AlienCPU and keeps its semantics hidden until after prediction.

From the repository root:

```bash
PYTHONPATH=src python demo/demo.py --seed 2026 --budget 20
```

or:

```bash
./scripts/run_demo.sh 2026 20
```

The script prints:

1. the unseen machine's public dimensions;
2. the fixed discovery budget;
3. selected black-box experiments;
4. inferred opcode hypotheses;
5. a held-out one-step prediction;
6. the actual hidden-simulator result;
7. a three-instruction held-out program prediction;
8. executable exact-match verification.

For the full reproducibility path, including tests, neural training/evaluation, system-identification evaluation, simulator benchmark, and demo:

```bash
./scripts/reproduce_mvp.sh
```

The seed-2026 contest demo uses the transparent system-identification predictor because it is currently the strongest and most interpretable model. The learned Transformer is evaluated separately and its results are reported without substituting or fabricating successes.
