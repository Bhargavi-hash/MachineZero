# Demo

Run the deterministic contest demo:

```bash
PYTHONPATH=src python demo/demo.py --seed 2026 --budget 20
```

or:

```bash
./scripts/run_demo.sh 2026 20
```

The script prints the hidden architecture summary, selected probes, learned hypotheses, a held-out one-step prediction, a three-instruction program prediction, and the actual simulator result. The demo does not reveal the ISA before prediction.
