"""Scenario mutation used by the BHUNTER fuzzer."""

from .exp_mutator import Mutator


def final_Mutator(seed_path):
    """Mutator used for the initial occlusion search."""
    return Mutator(seed_path)


def exp_Mutator(seed_path):
    """Mutator used when a queued scenario is executed again."""
    return Mutator(seed_path)
