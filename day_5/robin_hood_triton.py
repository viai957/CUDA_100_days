"""
Robin Hood Hashing - Triton Implementation

This implementation focuses on read-heavy workloads where the hash table
is built once and queried many times. Triton's lack of atomic operations
makes concurrent insertions challenging, so we use a CPU-side build phase
followed by GPU-accelerated lookups.
"""

import torch
import triton
import triton.language as tl
import time
import numpy as np

# Type configuration
KEY_TYPE = torch.int32
VALUE_TYPE = torch.int32
EMPTY_KEY = 0xFFFFFFFF
HASH_MULTIPLIER = 2654435761


@triton.jit
def hash_function(key, capacity):
    """Multiplicative hash function with Fibonacci constant"""
    return (key * HASH_MULTIPLIER) & (capacity - 1)


@triton.jit
def robin_hood_lookup_kernel(
    keys_ptr,           # Input: keys to lookup [num_keys]
    values_ptr,         # Output: values found [num_keys]
    found_ptr,          # Output: found flags [num_keys]
    table_keys_ptr,     # Hash table keys [capacity]
    table_values_ptr,   # Hash table values [capacity]
    table_psl_ptr,      # Hash table PSL [capacity]
    table_occupied_ptr, # Hash table occupied flags [capacity]
    num_keys,           # Number of keys to lookup
    capacity,           # Hash table capacity
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for Robin Hood hash table lookups.
    Each thread handles one key lookup.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_keys

    # Load keys to search for
    keys = tl.load(keys_ptr + offsets, mask=mask, other=EMPTY_KEY)

    # Initialize outputs
    values = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    found = tl.zeros([BLOCK_SIZE], dtype=tl.int1)

    # Process each key
    for i in range(BLOCK_SIZE):
        if offsets[i] < num_keys:
            key = keys[i]

            if key == EMPTY_KEY:
                continue

            # Calculate home position
            pos = hash_function(key, capacity)
            psl = 0
            key_found = False
            result_value = 0

            # Probe until we find the key or determine it doesn't exist
            for probe_iter in range(256):
                # Load slot data
                slot_key = tl.load(table_keys_ptr + pos)
                slot_value = tl.load(table_values_ptr + pos)
                slot_psl = tl.load(table_psl_ptr + pos)
                slot_occupied = tl.load(table_occupied_ptr + pos)

                # Empty slot - key not found
                if not slot_occupied:
                    break

                # Found the key
                if slot_key == key:
                    result_value = slot_value
                    key_found = True
                    break

                # PSL-based early termination
                if psl > slot_psl:
                    break

                # Move to next slot
                pos = (pos + 1) & (capacity - 1)
                psl += 1

            # Store results
            values[i] = result_value
            found[i] = key_found

    # Write results back
    tl.store(values_ptr + offsets, values, mask=mask)
    tl.store(found_ptr + offsets, found, mask=mask)


@triton.jit
def robin_hood_lookup_kernel_vectorized(
    keys_ptr,
    values_ptr,
    found_ptr,
    table_keys_ptr,
    table_values_ptr,
    table_psl_ptr,
    table_occupied_ptr,
    num_keys,
    capacity,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Vectorized Triton kernel for Robin Hood lookups.
    Uses SIMD operations for better performance.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_keys

    # Load keys
    keys = tl.load(keys_ptr + offsets, mask=mask, other=EMPTY_KEY)

    # Initialize home positions and PSLs
    positions = hash_function(keys, capacity)
    psls = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    values = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    found = tl.zeros([BLOCK_SIZE], dtype=tl.int1)
    active = mask & (keys != EMPTY_KEY)

    # Probe loop
    for probe_iter in range(256):
        if not tl.any(active):
            break

        # Load slot data for all active threads
        slot_keys = tl.load(table_keys_ptr + positions, mask=active, other=EMPTY_KEY)
        slot_values = tl.load(table_values_ptr + positions, mask=active, other=0)
        slot_psls = tl.load(table_psl_ptr + positions, mask=active, other=0)
        slot_occupied = tl.load(table_occupied_ptr + positions, mask=active, other=False)

        # Check conditions
        is_empty = ~slot_occupied
        is_match = slot_keys == keys
        psl_exceeded = psls > slot_psls

        # Update found keys
        newly_found = active & is_match
        values = tl.where(newly_found, slot_values, values)
        found = found | newly_found

        # Deactivate threads that found key, hit empty, or exceeded PSL
        active = active & ~(is_match | is_empty | psl_exceeded)

        # Move to next positions
        positions = (positions + 1) & (capacity - 1)
        psls = psls + 1

    # Write results
    tl.store(values_ptr + offsets, values, mask=mask)
    tl.store(found_ptr + offsets, found, mask=mask)


class RobinHoodHashTable:
    """
    Robin Hood Hash Table with Triton-accelerated lookups.

    This implementation uses CPU for building the hash table and
    GPU (via Triton) for fast batch lookups.
    """

    def __init__(self, capacity=65536, device='cuda'):
        assert capacity & (capacity - 1) == 0, "Capacity must be power of 2"

        self.capacity = capacity
        self.device = device
        self.size = 0

        # Allocate table on GPU
        self.table_keys = torch.full((capacity,), EMPTY_KEY, dtype=KEY_TYPE, device=device)
        self.table_values = torch.zeros(capacity, dtype=VALUE_TYPE, device=device)
        self.table_psl = torch.zeros(capacity, dtype=torch.uint8, device=device)
        self.table_occupied = torch.zeros(capacity, dtype=torch.bool, device=device)

    def hash(self, key):
        """Hash function"""
        return (key * HASH_MULTIPLIER) & (self.capacity - 1)

    def insert_cpu(self, keys, values):
        """
        CPU-side insertion using Robin Hood algorithm.
        For bulk insertions, this is done on CPU then transferred to GPU.
        """
        # Move to CPU for insertion
        table_keys = self.table_keys.cpu().numpy()
        table_values = self.table_values.cpu().numpy()
        table_psl = self.table_psl.cpu().numpy()
        table_occupied = self.table_occupied.cpu().numpy()

        keys = keys.cpu().numpy() if torch.is_tensor(keys) else keys
        values = values.cpu().numpy() if torch.is_tensor(values) else values

        for key, value in zip(keys, values):
            if key == EMPTY_KEY:
                continue

            pos = self.hash(key)
            curr_key = key
            curr_value = value
            curr_psl = 0

            while True:
                # Empty slot
                if not table_occupied[pos]:
                    table_keys[pos] = curr_key
                    table_values[pos] = curr_value
                    table_psl[pos] = curr_psl
                    table_occupied[pos] = True
                    self.size += 1
                    break

                # Key exists - update
                if table_keys[pos] == curr_key:
                    table_values[pos] = curr_value
                    break

                # Robin Hood displacement
                if curr_psl > table_psl[pos]:
                    # Swap
                    temp_key = table_keys[pos]
                    temp_value = table_values[pos]
                    temp_psl = table_psl[pos]

                    table_keys[pos] = curr_key
                    table_values[pos] = curr_value
                    table_psl[pos] = curr_psl

                    curr_key = temp_key
                    curr_value = temp_value
                    curr_psl = temp_psl

                pos = (pos + 1) & (self.capacity - 1)
                curr_psl += 1

        # Transfer back to GPU
        self.table_keys = torch.from_numpy(table_keys).to(self.device)
        self.table_values = torch.from_numpy(table_values).to(self.device)
        self.table_psl = torch.from_numpy(table_psl).to(self.device)
        self.table_occupied = torch.from_numpy(table_occupied).to(self.device)

    def lookup(self, keys, vectorized=True):
        """
        GPU-accelerated batch lookup using Triton.

        Args:
            keys: Tensor of keys to lookup
            vectorized: Use vectorized kernel (faster)

        Returns:
            values: Tensor of values found
            found: Boolean tensor indicating which keys were found
        """
        if not torch.is_tensor(keys):
            keys = torch.tensor(keys, dtype=KEY_TYPE, device=self.device)
        else:
            keys = keys.to(self.device)

        num_keys = keys.shape[0]
        values = torch.zeros(num_keys, dtype=VALUE_TYPE, device=self.device)
        found = torch.zeros(num_keys, dtype=torch.bool, device=self.device)

        # Choose block size based on hardware
        BLOCK_SIZE = 256
        grid = lambda meta: (triton.cdiv(num_keys, meta['BLOCK_SIZE']),)

        kernel = robin_hood_lookup_kernel_vectorized if vectorized else robin_hood_lookup_kernel

        kernel[grid](
            keys, values, found,
            self.table_keys,
            self.table_values,
            self.table_psl,
            self.table_occupied,
            num_keys,
            self.capacity,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return values, found

    def load_factor(self):
        """Return current load factor"""
        return self.size / self.capacity


def benchmark_robin_hood_triton():
    """Benchmark Robin Hood hash table with Triton"""
    print("=== Robin Hood Hash Table (Triton) Benchmark ===\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    if device == 'cpu':
        print("Warning: CUDA not available, skipping Triton benchmark")
        return

    # Test configuration
    capacity = 65536
    num_keys = 10000

    # Create hash table
    table = RobinHoodHashTable(capacity=capacity, device=device)

    # Generate test data
    keys = torch.arange(1, num_keys + 1, dtype=KEY_TYPE)
    values = keys * 2

    # Test 1: Bulk insertion (CPU)
    print("Test 1: Bulk Insertion (CPU-side)")
    start = time.time()
    table.insert_cpu(keys, values)
    insert_time = (time.time() - start) * 1000
    print(f"Inserted {num_keys} keys in {insert_time:.2f} ms")
    print(f"Load factor: {table.load_factor():.2%}\n")

    # Test 2: Batch lookup (standard kernel)
    print("Test 2: Batch Lookup (Standard Kernel)")

    # Warmup
    for _ in range(10):
        table.lookup(keys, vectorized=False)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(100):
        result_values, result_found = table.lookup(keys, vectorized=False)
    torch.cuda.synchronize()
    lookup_time = (time.time() - start) * 1000 / 100

    print(f"Looked up {num_keys} keys in {lookup_time:.2f} ms")

    # Verify correctness
    assert result_found.all(), "Not all keys found!"
    assert (result_values == values.to(device)).all(), "Values don't match!"
    print("Verification: PASSED\n")

    # Test 3: Batch lookup (vectorized kernel)
    print("Test 3: Batch Lookup (Vectorized Kernel)")

    # Warmup
    for _ in range(10):
        table.lookup(keys, vectorized=True)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(100):
        result_values, result_found = table.lookup(keys, vectorized=True)
    torch.cuda.synchronize()
    lookup_vec_time = (time.time() - start) * 1000 / 100

    print(f"Looked up {num_keys} keys in {lookup_vec_time:.2f} ms")
    print(f"Speedup: {lookup_time / lookup_vec_time:.2f}x over standard\n")

    # Verify correctness
    assert result_found.all(), "Not all keys found!"
    assert (result_values == values.to(device)).all(), "Values don't match!"
    print("Verification: PASSED\n")

    # Test 4: Lookup performance vs PyTorch dict
    print("Test 4: Comparison with Python dict")

    # Build Python dict
    py_dict = {k.item(): v.item() for k, v in zip(keys, values)}

    # Lookup with Python dict
    start = time.time()
    for _ in range(100):
        results = [py_dict[k.item()] for k in keys]
    py_lookup_time = (time.time() - start) * 1000 / 100

    print(f"Python dict lookup: {py_lookup_time:.2f} ms")
    print(f"Triton speedup: {py_lookup_time / lookup_vec_time:.2f}x\n")

    # Test 5: PSL statistics
    print("Test 5: PSL Statistics")
    psl_values = table.table_psl[table.table_occupied].cpu().numpy()

    if len(psl_values) > 0:
        avg_psl = psl_values.mean()
        max_psl = psl_values.max()
        print(f"Average PSL: {avg_psl:.2f}")
        print(f"Maximum PSL: {max_psl}")
        print("(Lower PSL values indicate better performance)\n")

    print("=== All Triton tests passed! ===")


if __name__ == "__main__":
    benchmark_robin_hood_triton()
