"""
Hash Table Lookup (Triton): parallel open-addressing table probe
Math:
  - Build (CPU): for each (key k, value v), insert into table T of size C (power of two)
        h0 = hash(k) & (C - 1)
        hi = (h0 + i) & (C - 1) until EMPTY slot, then
        T.keys[hi] = k, T.vals[hi] = v.
  - Lookup (GPU/Triton): for each query key q, probe the same sequence and
        return the value at the first slot where T.keys[hi] == q, or report
        "not found" when hitting an EMPTY slot.
Inputs / Outputs:
  - Table (built on CPU, stored on GPU): table_keys[C], table_vals[C]
  - Queries: query_keys[M] -> out_vals[M], found[M] (0/1 flags)
Assumptions:
  - Keys and values are int32.
  - Table capacity C is a power of two and chosen so that load factor <= 0.5.
  - No deletions, no concurrent writes during lookup; probes are read-only.
Parallel Strategy:
  - One Triton program processes a block of query keys with coalesced memory
    access to both the query array and the hash table.
Mixed Precision Policy:
  - All integer math; no mixed precision.
Distributed Hooks:
  - Single-GPU example; could be wrapped in data-parallel sharding if desired.
Complexity:
  - Expected O(M) lookups with constant expected probe length at load factor <= 0.5.
Test Vectors:
  - Keys: key_i = 2*i + 1, values: val_i = 10*i for i in [0, N).
  - Queries: inserted keys (must be found) plus a small number of missing keys.
"""

import math
from typing import Tuple

import torch
import triton
import triton.language as tl


# Sentinel for empty slots; must never be used as a real key
EMPTY_KEY = -2 ** 31  # Matches INT_MIN in C


def _next_power_of_two(x: int) -> int:
    """Return the next power of two >= x."""
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def build_hash_table_cpu(keys: torch.Tensor, values: torch.Tensor, load_factor: float = 0.5
                         ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Build an open-addressing hash table on CPU using linear probing.

    Args:
        keys: 1D int32 tensor on CPU with unique keys.
        values: 1D int32 tensor on CPU with corresponding values.
        load_factor: Target max load factor (e.g., 0.5).

    Returns:
        table_keys: 1D int32 tensor of length capacity (CPU).
        table_vals: 1D int32 tensor of length capacity (CPU).
        capacity: Table capacity (power of two).
        mask: capacity - 1 for bitwise modulo.
    """
    assert keys.device.type == "cpu" and values.device.type == "cpu", "Build must run on CPU tensors"
    assert keys.dtype == torch.int32 and values.dtype == torch.int32, "keys/values must be int32"
    assert keys.shape == values.shape and keys.dim() == 1, "keys and values must be 1D and same length"

    n = keys.numel()
    assert n > 0, "Need at least one key"

    # Choose capacity so that load_factor <= requested threshold, and make it a power of two
    min_capacity = int(math.ceil(n / max(load_factor, 1e-3)))
    capacity = _next_power_of_two(min_capacity)
    mask = capacity - 1

    table_keys = torch.full((capacity,), EMPTY_KEY, dtype=torch.int32)
    table_vals = torch.zeros((capacity,), dtype=torch.int32)

    for i in range(n):
        k = int(keys[i].item())
        v = int(values[i].item())
        # Simple integer hash; masked into [0, capacity)
        h = (k * 9973) & mask
        idx = h
        # Linear probing
        for _ in range(capacity):
            if table_keys[idx] == EMPTY_KEY or table_keys[idx] == k:
                table_keys[idx] = k
                table_vals[idx] = v
                break
            idx = (idx + 1) & mask
        else:
            raise RuntimeError("Hash table over-full during CPU build; increase capacity or lower load factor")

    return table_keys, table_vals, capacity, mask


@triton.jit
def hash_table_lookup_kernel(
    table_keys_ptr,  # int32[C]
    table_vals_ptr,  # int32[C]
    query_keys_ptr,  # int32[M]
    out_vals_ptr,    # int32[M]
    found_ptr,       # int8[M]
    num_queries,     # int32 scalar
    capacity,        # int32 scalar (power of two)
    mask,            # int32 scalar = capacity - 1
    BLOCK_SIZE: tl.constexpr,
    MAX_STEPS: tl.constexpr,
):
    """Triton kernel: probe hash table for a block of query keys.

    Each program handles BLOCK_SIZE queries. Probing is linear with a fixed
    MAX_STEPS bound for safety; at typical load factors most queries finish
    quickly.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_queries = offsets < num_queries

    # Load query keys
    q_keys = tl.load(query_keys_ptr + offsets, mask=mask_queries, other=0)

    # Initialize state
    cap_mask = mask
    empty_key = tl.full_like(q_keys, EMPTY_KEY)

    # Simple integer hash: h = (k * 9973) & mask
    h = (q_keys * 9973) & cap_mask
    idx = h

    found = tl.zeros_like(q_keys, dtype=tl.int1)
    vals = tl.zeros_like(q_keys, dtype=tl.int32)

    for _ in range(MAX_STEPS):
        active = mask_queries & (~found)
        if not tl.any(active):
            break

        slot_keys = tl.load(table_keys_ptr + idx, mask=active, other=EMPTY_KEY)
        slot_vals = tl.load(table_vals_ptr + idx, mask=active, other=0)

        is_match = active & (slot_keys == q_keys)
        vals = tl.where(is_match, slot_vals, vals)
        found = found | is_match

        # Early-exit for keys whose probe chain hit EMPTY_KEY
        hit_empty = active & (slot_keys == empty_key)
        found = found | hit_empty  # mark as done (not found) for those

        # Advance probe index for still-active, not-yet-done lanes
        still_searching = active & (~hit_empty) & (~is_match)
        idx = tl.where(still_searching, (idx + 1) & cap_mask, idx)

    # Store outputs
    tl.store(out_vals_ptr + offsets, vals, mask=mask_queries)
    tl.store(found_ptr + offsets, found.to(tl.int8), mask=mask_queries)


def hash_table_lookup_triton(
    table_keys: torch.Tensor,
    table_vals: torch.Tensor,
    query_keys: torch.Tensor,
    capacity: int,
    mask_val: int,
    block_size: int = 256,
    max_steps: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run Triton lookup kernel on the given hash table and queries.

    Args:
        table_keys: 1D int32 CUDA tensor of length capacity.
        table_vals: 1D int32 CUDA tensor of length capacity.
        query_keys: 1D int32 CUDA tensor of length M.
        capacity: Table capacity (power of two).
        mask_val: capacity - 1.
        block_size: Triton block size for queries.
        max_steps: Max number of probe steps per query.

    Returns:
        out_vals: 1D int32 CUDA tensor of length M.
        found: 1D int8 CUDA tensor of length M (0/1 flags).
    """
    assert table_keys.device.type == "cuda" and table_vals.device.type == "cuda", "Table must be on CUDA device"
    assert query_keys.device.type == "cuda", "Queries must be on CUDA device"
    assert table_keys.dtype == torch.int32 and table_vals.dtype == torch.int32, "Table must be int32"
    assert query_keys.dtype == torch.int32, "Queries must be int32"
    assert table_keys.numel() == capacity and table_vals.numel() == capacity, "Capacity mismatch"

    num_queries = query_keys.numel()
    out_vals = torch.zeros_like(query_keys)
    found = torch.zeros(num_queries, dtype=torch.int8, device=query_keys.device)

    grid = (math.ceil(num_queries / block_size),)

    hash_table_lookup_kernel[grid](
        table_keys,
        table_vals,
        query_keys,
        out_vals,
        found,
        num_queries,
        capacity,
        mask_val,
        BLOCK_SIZE=block_size,
        MAX_STEPS=max_steps,
    )

    return out_vals, found


def test_hash_table_triton(N: int = 1 << 17) -> None:
    """End-to-end test: build table on CPU, probe on GPU with Triton.

    Mirrors the CUDA test harness in day_4/hash_table.cu.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available, skipping Triton hash table test")
        return

    torch.manual_seed(0)

    # Generate deterministic keys and values on CPU
    keys_cpu = (torch.arange(N, dtype=torch.int32) * 2 + 1)
    vals_cpu = (torch.arange(N, dtype=torch.int32) * 10)

    # Build hash table on CPU
    table_keys_cpu, table_vals_cpu, capacity, mask_val = build_hash_table_cpu(keys_cpu, vals_cpu, load_factor=0.5)

    # Prepare queries: existing keys + a few missing ones
    extra_misses = 8
    num_queries = N + extra_misses

    query_keys_cpu = torch.empty(num_queries, dtype=torch.int32)
    query_keys_cpu[:N] = keys_cpu
    for i in range(extra_misses):
        query_keys_cpu[N + i] = -2 * (i + 1)

    # Move table and queries to GPU
    table_keys_gpu = table_keys_cpu.to(device)
    table_vals_gpu = table_vals_cpu.to(device)
    query_keys_gpu = query_keys_cpu.to(device)

    # Warmup
    for _ in range(5):
        _ = hash_table_lookup_triton(table_keys_gpu, table_vals_gpu, query_keys_gpu, capacity, mask_val)
    torch.cuda.synchronize()

    # Timed run
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    out_vals_gpu, found_gpu = hash_table_lookup_triton(
        table_keys_gpu,
        table_vals_gpu,
        query_keys_gpu,
        capacity,
        mask_val,
        block_size=256,
        max_steps=64,
    )
    end_event.record()

    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)

    print(f"Triton Hash Table Lookup - elapsed time: {elapsed_ms:.3f} ms")

    # Verify correctness
    out_vals = out_vals_gpu.cpu()
    found = found_gpu.cpu()

    num_errors = 0

    # Existing keys must be found with correct values
    for i in range(N):
        if found[i].item() != 1:
            if num_errors < 8:
                print(f"ERROR: key {int(query_keys_cpu[i])} not found (index {i})")
            num_errors += 1
            continue
        expected = vals_cpu[i].item()
        if out_vals[i].item() != expected:
            if num_errors < 8:
                print(
                    f"ERROR: wrong value for key {int(query_keys_cpu[i])} at index {i}: "
                    f"got {int(out_vals[i].item())}, expected {expected}"
                )
            num_errors += 1

    # Missing keys must be reported as not found
    for i in range(N, num_queries):
        if found[i].item() != 0:
            if num_errors < 8:
                print(f"ERROR: missing key {int(query_keys_cpu[i])} was reported as found (index {i})")
            num_errors += 1

    if num_errors == 0:
        print("Triton hash table lookup: result OK")
    else:
        print(f"Triton hash table lookup: {num_errors} errors detected")


if __name__ == "__main__":
    test_hash_table_triton(1 << 17)
