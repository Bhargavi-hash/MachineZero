from __future__ import annotations

from machinezero.models.prediction import predict_state


def predict_program(model, context, state, program, word_bits, num_registers, device="cpu"):
    """Predict a fixed instruction sequence without consulting the hidden simulator."""
    cur = state.clone()
    for ins in program:
        cur = predict_state(model, context, cur, ins, word_bits, num_registers, device)
    return cur
