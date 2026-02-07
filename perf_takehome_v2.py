"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.const_vectorized_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def scratch_const_vectorized(self, val, name=None, length=VLEN):
        if val not in self.const_vectorized_map:
            addr = self.alloc_scratch(name, length=length)
            self.add("load", ("const", addr, val))
            self.add("valu", ("vbroadcast", addr, addr))
            self.const_vectorized_map[val] = addr
        return self.const_vectorized_map[val]

    def form_debug_compare_vector(self, addr, round, debug_range, *key_suffix):
        """Form a debug vcompare slot for vector comparison."""
        return ("debug", ("vcompare", addr, [(round, bi, *key_suffix) for bi in debug_range]))

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1, f"const_{val1}"))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3, f"const_{val3}"))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, [(round, i, "hash_stage", hi)])))

        return slots

    def build_hash_vectorized(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []
        i_offset = i * VLEN
        debug_range = range(i_offset, i_offset + VLEN)

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(
                ("valu", (op1, tmp1, val_hash_addr, self.scratch_const_vectorized(val1, f"const_hash_{val1}")))
            )
            slots.append(
                ("valu", (op3, tmp2, val_hash_addr, self.scratch_const_vectorized(val3, f"const_hash_{val3}")))
            )
            slots.append(("valu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(self.form_debug_compare_vector(val_hash_addr, round, debug_range, "hash_stage", hi))

        return slots

    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        """
        Like reference_kernel2 but building actual instructions.
        Vectorized implementation using vector ALU and load/store + store values and indices in scratchpad
        """
        tmp1 = self.alloc_scratch("tmp1", length=VLEN)
        tmp2 = self.alloc_scratch("tmp2", length=VLEN)
        tmp3 = self.alloc_scratch("tmp3", length=VLEN)
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const_vectorized = self.scratch_const_vectorized(0, name="zero_const_vectorized")
        one_const_vectorized = self.scratch_const_vectorized(1, name="one_const_vectorized")
        two_const_vectorized = self.scratch_const_vectorized(2, name="two_const_vectorized")

        forest_values_p_vectorized = self.alloc_scratch("forest_values_p_vectorized", length=VLEN)
        self.add("valu", ("vbroadcast", forest_values_p_vectorized, self.scratch["forest_values_p"]))

        n_nodes_vectorized = self.alloc_scratch("n_nodes_vectorized", length=VLEN)
        self.add("valu", ("vbroadcast", n_nodes_vectorized, self.scratch["n_nodes"]))

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots
        num_chunks = batch_size // VLEN
        tmp_addr_scalar = self.alloc_scratch("tmp_addr_scalar")

        # Allocate index and val
        idx_full_vec = self.alloc_scratch("idx_full_vec", length=batch_size)
        for i in range(num_chunks):
            i_offset = i * VLEN
            i_offset_addr = self.scratch_const(i_offset, f"const_{i}")
            body.append(("alu", ("+", tmp_addr_scalar, self.scratch["inp_indices_p"], i_offset_addr)))
            body.append(("load", ("vload", idx_full_vec + i_offset, tmp_addr_scalar)))
        val_full_vec = self.alloc_scratch("val_full_vec", length=batch_size)
        for i in range(num_chunks):
            i_offset = i * VLEN
            i_offset_addr = self.scratch_const(i_offset, f"const_{i}")
            body.append(("alu", ("+", tmp_addr_scalar, self.scratch["inp_values_p"], i_offset_addr)))
            body.append(("load", ("vload", val_full_vec + i_offset, tmp_addr_scalar)))

        # Allocate temporary node value
        tmp_node_val = self.alloc_scratch("tmp_node_val", length=VLEN)
        tmp_addr = self.alloc_scratch("tmp_addr", length=VLEN)

        for round in range(rounds):
            for i in range(num_chunks):
                i_offset = i * VLEN
                debug_range = range(i_offset, i_offset + VLEN)
                i_offset_addr = self.scratch_const(i_offset, f"const_{i}")

                # idx = mem[inp_indices_p + i]
                tmp_idx = idx_full_vec + i_offset
                body.append(self.form_debug_compare_vector(tmp_idx, round, debug_range, "idx"))

                # val = mem[inp_values_p + i]
                tmp_val = val_full_vec + i_offset
                body.append(self.form_debug_compare_vector(tmp_val, round, debug_range, "val"))

                # node_val = mem[forest_values_p + idx]
                body.append(("valu", ("+", tmp_addr, forest_values_p_vectorized, tmp_idx)))
                for vi in range(VLEN):
                    body.append(("load", ("load_offset", tmp_node_val, tmp_addr, vi)))
                body.append(self.form_debug_compare_vector(tmp_node_val, round, debug_range, "node_val"))

                # val = myhash(val ^ node_val)
                body.append(("valu", ("^", tmp_val, tmp_val, tmp_node_val)))
                body.extend(self.build_hash_vectorized(tmp_val, tmp1, tmp2, round, i))
                body.append(self.form_debug_compare_vector(tmp_val, round, debug_range, "hashed_val"))

                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("valu", ("%", tmp1, tmp_val, two_const_vectorized)))
                body.append(("valu", ("==", tmp1, tmp1, zero_const_vectorized)))
                body.append(("flow", ("vselect", tmp3, tmp1, one_const_vectorized, two_const_vectorized)))
                body.append(("valu", ("*", tmp_idx, tmp_idx, two_const_vectorized)))
                body.append(("valu", ("+", tmp_idx, tmp_idx, tmp3)))
                body.append(self.form_debug_compare_vector(tmp_idx, round, debug_range, "next_idx"))

                # idx = 0 if idx >= n_nodes else idx
                body.append(("valu", ("<", tmp1, tmp_idx, n_nodes_vectorized)))
                body.append(("flow", ("vselect", tmp_idx, tmp1, tmp_idx, zero_const_vectorized)))
                body.append(self.form_debug_compare_vector(tmp_idx, round, debug_range, "wrapped_idx"))

            # updating indices in memory is not required
            # updating input values in memory

        for i in range(num_chunks):
            i_offset = i * VLEN
            i_offset_addr = self.scratch_const(i_offset, f"const_{i}")
            body.append(("alu", ("+", tmp_addr_scalar, self.scratch["inp_values_p"], i_offset_addr)))
            body.append(("store", ("vstore", tmp_addr_scalar, val_full_vec + i_offset)))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})


BASELINE = 147734


def do_kernel_test(
    forest_height: int, rounds: int, batch_size: int, seed: int = 123, trace: bool = False, prints: bool = False
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES, value_trace=value_trace, trace=trace)
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
