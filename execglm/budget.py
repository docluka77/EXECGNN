"""Exact optimizer-update budgets; legacy epoch mode remains available."""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Budget:
    epochs: int
    updates: int
    batches_per_epoch: int


def resolve_budget(n_items, batch, epochs, min_steps=0, max_steps=0):
    if n_items <= 0 or batch <= 0 or epochs <= 0:
        raise ValueError('training items, batch and epochs must be positive')
    if min_steps < 0 or max_steps < 0:
        raise ValueError('update limits must be nonnegative')
    if min_steps and max_steps:
        raise ValueError('min_steps and max_steps are mutually exclusive')
    per_epoch = math.ceil(n_items / batch)
    if max_steps:
        return Budget(math.ceil(max_steps / per_epoch), max_steps, per_epoch)
    if min_steps and epochs * per_epoch < min_steps:
        epochs = math.ceil(min_steps / per_epoch)
    return Budget(epochs, epochs * per_epoch, per_epoch)
