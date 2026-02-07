"""
Analyze kernel instruction mix and compute per-engine lower-bound cycle
limitations (= ops // SLOT_LIMITS). The theoretical minimum cycle count
with perfect VLIW packing is max(limitations) across non-debug engines.
"""

import random
from perf_takehome import KernelBuilder
from problem import SLOT_LIMITS, Input, Tree

FOREST_HEIGHT = 10
BATCH_SIZE = 256
ROUNDS = 16


def collect_key_to_elements(instrs):
    unique_keys = set()
    for instr in instrs:
        unique_keys.update(instr.keys())
    key_to_elements = {k: [] for k in unique_keys}
    for instr in instrs:
        for k in unique_keys:
            if k in instr:
                key_to_elements[k].append(instr[k])
    return key_to_elements


def num_operations_per_key(key_to_elements):
    """Total number of slot operations for each engine."""
    return {
        k: sum(len(slots) for slots in v)
        for k, v in key_to_elements.items()
    }


def calculate_limitations(key_num_ops):
    """Lower-bound cycles per engine = ops // slots_per_cycle."""
    return {
        k: v // SLOT_LIMITS[k] for k, v in key_num_ops.items()
    }


def analyze(kb):
    key_to_elements = collect_key_to_elements(kb.instrs)
    key_num_ops = num_operations_per_key(key_to_elements)
    limitations = calculate_limitations(key_num_ops)

    print("=== Operation counts ===")
    for k in sorted(key_num_ops.keys()):
        print(
            f"  {k:>6}: {key_num_ops[k]:>6}"
            f"  (slots: {SLOT_LIMITS[k]})"
        )

    print()
    print("=== Limitations (lower bound cycles per engine) ===")
    for k in sorted(limitations.keys(), key=lambda x: -limitations[x]):
        print(f"  {k:>6}: {limitations[k]:>6}")

    non_debug = {
        k: v for k, v in limitations.items() if k != "debug"
    }
    theoretical_min = max(non_debug.values())
    bottleneck = max(non_debug, key=non_debug.get)

    print()
    print(f"Total instructions: {len(kb.instrs)}")
    print(f"Scratch used: {kb.scratch_ptr} / 1536")
    print(f"Theoretical min: {theoretical_min}  (bottleneck: {bottleneck})")

    return key_num_ops, limitations


if __name__ == "__main__":
    random.seed(123)
    forest = Tree.generate(FOREST_HEIGHT)
    inp = Input.generate(forest, BATCH_SIZE, ROUNDS)

    kb = KernelBuilder()
    kb.build_kernel(
        FOREST_HEIGHT, len(forest.values), len(inp.indices), ROUNDS
    )

    analyze(kb)
