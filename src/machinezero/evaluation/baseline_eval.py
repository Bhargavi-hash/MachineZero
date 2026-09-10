from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from machinezero.aliencpu.generator import generate_architecture
from machinezero.aliencpu.simulator import AlienCPU
from machinezero.data.splits import make_splits
from machinezero.models.baselines import NoOpPredictor, RandomStatePredictor

from .evaluate import sample_query
from .metrics import compare_states


def evaluate(queries_per_arch: int = 40):
    splits = make_splits(202600, 48, 8, 8)
    totals = {
        "random_state": {"exact": 0.0, "register_accuracy": 0.0, "flag_accuracy": 0.0},
        "no_op": {"exact": 0.0, "register_accuracy": 0.0, "flag_accuracy": 0.0},
    }
    count = 0
    for seed in splits.test:
        spec = generate_architecture(seed)
        cpu = AlienCPU(spec)
        rng = random.Random(seed * 211)
        random_model = RandomStatePredictor(spec.word_bits, spec.num_registers, seed)
        no_op = NoOpPredictor(spec.word_bits)
        for _ in range(queries_per_arch):
            state, ins = sample_query(rng, spec)
            actual = cpu.step(state, ins)
            predictions = {
                "random_state": random_model.predict(),
                "no_op": no_op.predict(state),
            }
            for name, pred in predictions.items():
                metrics = compare_states(pred, actual)
                for key in totals[name]:
                    totals[name][key] += metrics[key]
            count += 1

    rows = [
        {
            "model": name,
            "architectures": len(splits.test),
            "queries": count,
            **{key: value / count for key, value in sums.items()},
        }
        for name, sums in totals.items()
    ]
    for row in rows:
        print(row)
    return {"held_out_test_seeds": list(splits.test), "results": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/baselines.json")
    parser.add_argument("--queries-per-arch", type=int, default=40)
    args = parser.parse_args()
    results = evaluate(args.queries_per_arch)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
