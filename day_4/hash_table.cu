/*
 * GpuHashTable: parallel open-addressing hash table build + query
 * Math: For each key k with value v, we insert into a table T of size C (power of two)
 *       using multiplicative hashing and linear probing:
 *           h0 = (k * 2654435761u) & (C - 1)
 *           hi = (h0 + i) & (C - 1) until an EMPTY slot is found, then
 *           T.keys[hi] = k, T.vals[hi] = v.
 *       Lookup for query key q probes the same sequence and returns the first
 *       matching key or reports "not found" when hitting an EMPTY slot.
 * Inputs / Outputs:
 *   - Build: input_keys[N], input_vals[N]  -> hash table (keys[C], vals[C])
 *   - Query: query_keys[M] -> out_vals[M], found[M] (0/1 flags)
 *   All arrays are contiguous int32 buffers in device or host memory.
 * Assumptions:
 *   - Keys are 32-bit signed integers and never equal to EMPTY_KEY (INT_MIN).
 *   - Table capacity C is a power of two and at least 2x the number of inserted keys.
 *   - No deletions; open addressing with linear probing is sufficient.
 *   - Single GPU, no inter-GPU communication.
 * Parallel Strategy:
 *   - One CUDA thread per (key, value) pair during build; each thread performs a
 *     private probe sequence using atomicCAS on the key array to claim a slot.
 *   - One CUDA thread per query key during lookup; each thread performs a read-only
 *     probe sequence until it finds its key or an EMPTY slot.
 * Mixed Precision Policy:
 *   - All data is 32-bit integers; no mixed-precision math is required.
 * Distributed Hooks:
 *   - This example is single-GPU only. NCCL or multi-GPU sharding could be added
 *     around the build/query phases if needed.
 * Complexity:
 *   - Expected O(N) inserts and O(M) lookups assuming constant expected probe length
 *     at load factor <= 0.5. Bytes moved: O((N + M) * sizeof(int)).
 * Test Vectors:
 *   - Deterministic keys: key_i = 2*i + 1, val_i = 10*i for i in [0, N).
 *   - Queries: the N inserted keys (must be found) plus a small number of
 *     non-existent keys (must report not found).
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <assert.h>
#include <sys/time.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_common.cuh"

// Type aliases for hash table keys and values
typedef int32_t KeyType;
typedef int32_t ValueType;

// Sentinel for empty slots; must never appear as a real key
static const KeyType EMPTY_KEY = INT_MIN;

struct DeviceHashTable {
    KeyType* keys;   // length = capacity
    ValueType* vals; // length = capacity
    int capacity;    // power of two
    int mask;        // capacity - 1, for cheap modulo
};

// Simple multiplicative hash mapped into [0, capacity)
__host__ __device__ inline int hash_int(KeyType k, int mask) {
    // Cast to unsigned to get well-defined overflow behavior
    uint32_t x = static_cast<uint32_t>(k);
    x *= 2654435761u; // Knuth multiplicative hash constant
    return static_cast<int>(x & static_cast<uint32_t>(mask));
}

// Initialize all table slots to EMPTY_KEY
__global__ void init_table(DeviceHashTable table) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < table.capacity) {
        table.keys[tid] = EMPTY_KEY;
        table.vals[tid] = 0;
    }
}

// Kernel: build hash table from input key/value pairs using open addressing
__global__ void build_table_kernel(DeviceHashTable table,
                                   const KeyType* __restrict__ input_keys,
                                   const ValueType* __restrict__ input_vals,
                                   int num_items) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_items) {
        return;
    }

    KeyType key = input_keys[tid];
    ValueType val = input_vals[tid];

    // Probe sequence starting from hashed index
    int idx = hash_int(key, table.mask);

    // We allow at most capacity probes; with load factor <= 0.5 this is
    // extremely unlikely to be hit in practice, but it guarantees termination.
    for (int step = 0; step < table.capacity; ++step) {
        int slot = (idx + step) & table.mask;
        KeyType* slot_ptr = &table.keys[slot];

        KeyType prev = atomicCAS(slot_ptr, EMPTY_KEY, key);

        if (prev == EMPTY_KEY || prev == key) {
            // Either we claimed an empty slot, or the key was already present.
            // In both cases, we can safely write the value.
            table.vals[slot] = val;
            return;
        }
        // Otherwise, another thread owns this slot with a different key; continue probing.
    }
    // If we get here, the table is over-full; for this educational example
    // we simply drop the item. In production, we would track and report this.
}

// Kernel: look up query keys in the hash table
__global__ void lookup_kernel(DeviceHashTable table,
                              const KeyType* __restrict__ query_keys,
                              ValueType* __restrict__ out_vals,
                              uint8_t* __restrict__ out_found,
                              int num_queries) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_queries) {
        return;
    }

    KeyType key = query_keys[tid];
    int idx = hash_int(key, table.mask);

    ValueType result = 0;
    uint8_t found = 0;

    for (int step = 0; step < table.capacity; ++step) {
        int slot = (idx + step) & table.mask;
        KeyType slot_key = table.keys[slot];

        if (slot_key == key) {
            result = table.vals[slot];
            found = 1;
            break;
        }
        if (slot_key == EMPTY_KEY) {
            // Once we hit an empty slot, the key does not exist in the table
            break;
        }
    }

    out_vals[tid] = result;
    out_found[tid] = found;
}

// Host-side test harness mirroring the C hash table examples
void test_gpu_hash_table(int num_items) {
    printf("=== GPU Hash Table Test (N = %d) ===\n", num_items);

    // Choose capacity as the next power of two >= 2 * num_items
    int capacity = 1;
    while (capacity < 2 * num_items) {
        capacity <<= 1;
    }
    int mask = capacity - 1;

    printf("Table capacity: %d (load factor <= %.2f)\n", capacity,
           (double)num_items / (double)capacity);

    // Allocate host input arrays
    KeyType* h_keys = (KeyType*)malloc(sizeof(KeyType) * num_items);
    ValueType* h_vals = (ValueType*)malloc(sizeof(ValueType) * num_items);

    if (!h_keys || !h_vals) {
        fprintf(stderr, "Host allocation failed\n");
        exit(1);
    }

    // Deterministic test data: keys are odd integers, values are simple multiples
    for (int i = 0; i < num_items; ++i) {
        h_keys[i] = (KeyType)(2 * i + 1);
        h_vals[i] = (ValueType)(10 * i);
    }

    // Prepare queries: first all inserted keys, then a few missing keys
    int extra_misses = 8;
    int num_queries = num_items + extra_misses;

    KeyType* h_query_keys = (KeyType*)malloc(sizeof(KeyType) * num_queries);
    ValueType* h_query_vals = (ValueType*)malloc(sizeof(ValueType) * num_queries);
    uint8_t* h_found = (uint8_t*)malloc(sizeof(uint8_t) * num_queries);

    if (!h_query_keys || !h_query_vals || !h_found) {
        fprintf(stderr, "Host allocation for queries failed\n");
        exit(1);
    }

    // First N queries are existing keys
    for (int i = 0; i < num_items; ++i) {
        h_query_keys[i] = h_keys[i];
    }
    // Remaining queries are keys that are guaranteed to be absent
    for (int i = 0; i < extra_misses; ++i) {
        h_query_keys[num_items + i] = (KeyType)(-2 * (i + 1));
    }

    // Allocate device memory for inputs, table, and outputs
    KeyType *d_input_keys = nullptr, *d_table_keys = nullptr, *d_query_keys = nullptr;
    ValueType *d_input_vals = nullptr, *d_table_vals = nullptr, *d_query_vals = nullptr;
    uint8_t* d_found = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_input_keys, sizeof(KeyType) * num_items));
    CUDA_CHECK(cudaMalloc((void**)&d_input_vals, sizeof(ValueType) * num_items));
    CUDA_CHECK(cudaMalloc((void**)&d_query_keys, sizeof(KeyType) * num_queries));
    CUDA_CHECK(cudaMalloc((void**)&d_query_vals, sizeof(ValueType) * num_queries));
    CUDA_CHECK(cudaMalloc((void**)&d_found, sizeof(uint8_t) * num_queries));

    CUDA_CHECK(cudaMalloc((void**)&d_table_keys, sizeof(KeyType) * capacity));
    CUDA_CHECK(cudaMalloc((void**)&d_table_vals, sizeof(ValueType) * capacity));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_input_keys, h_keys, sizeof(KeyType) * num_items, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input_vals, h_vals, sizeof(ValueType) * num_items, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_query_keys, h_query_keys, sizeof(KeyType) * num_queries, cudaMemcpyHostToDevice));

    DeviceHashTable table;
    table.keys = d_table_keys;
    table.vals = d_table_vals;
    table.capacity = capacity;
    table.mask = mask;

    // Launch configuration helpers from cuda_common.cuh
    int block_size_build = calculate_optimal_block_size(num_items);
    int grid_size_build = calculate_grid_size(num_items, block_size_build);

    int block_size_table = calculate_optimal_block_size(capacity);
    int grid_size_table = calculate_grid_size(capacity, block_size_table);

    int block_size_lookup = calculate_optimal_block_size(num_queries);
    int grid_size_lookup = calculate_grid_size(num_queries, block_size_lookup);

    cudaEvent_t start_build, stop_build, start_lookup, stop_lookup;
    CUDA_CHECK(cudaEventCreate(&start_build));
    CUDA_CHECK(cudaEventCreate(&stop_build));
    CUDA_CHECK(cudaEventCreate(&start_lookup));
    CUDA_CHECK(cudaEventCreate(&stop_lookup));

    // Initialize table
    init_table<<<grid_size_table, block_size_table>>>(table);
    CUDA_CHECK_KERNEL();

    // Build phase timing
    CUDA_CHECK(cudaEventRecord(start_build));
    build_table_kernel<<<grid_size_build, block_size_build>>>(table, d_input_keys, d_input_vals, num_items);
    CUDA_CHECK(cudaEventRecord(stop_build));
    CUDA_CHECK_KERNEL();

    float ms_build = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_build, start_build, stop_build));

    // Lookup phase timing
    CUDA_CHECK(cudaEventRecord(start_lookup));
    lookup_kernel<<<grid_size_lookup, block_size_lookup>>>(table, d_query_keys, d_query_vals, d_found, num_queries);
    CUDA_CHECK(cudaEventRecord(stop_lookup));
    CUDA_CHECK_KERNEL();

    float ms_lookup = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_lookup, start_lookup, stop_lookup));

    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_query_vals, d_query_vals, sizeof(ValueType) * num_queries, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_found, d_found, sizeof(uint8_t) * num_queries, cudaMemcpyDeviceToHost));

    // Verify correctness
    int num_errors = 0;

    // Existing keys must be found with correct values
    for (int i = 0; i < num_items; ++i) {
        if (h_found[i] != 1) {
            if (num_errors < 8) {
                printf("ERROR: key %d not found (index %d)\n", (int)h_query_keys[i], i);
            }
            ++num_errors;
            continue;
        }
        ValueType expected = h_vals[i];
        if (h_query_vals[i] != expected) {
            if (num_errors < 8) {
                printf("ERROR: wrong value for key %d at index %d: got %d, expected %d\n",
                       (int)h_query_keys[i], i, (int)h_query_vals[i], (int)expected);
            }
            ++num_errors;
        }
    }

    // Missing keys must be reported as not found
    for (int i = num_items; i < num_queries; ++i) {
        if (h_found[i] != 0) {
            if (num_errors < 8) {
                printf("ERROR: missing key %d was reported as found (index %d)\n",
                       (int)h_query_keys[i], i);
            }
            ++num_errors;
        }
    }

    if (num_errors == 0) {
        printf("Hash table build + lookup: result OK\n");
    } else {
        printf("Hash table build + lookup: %d errors detected\n", num_errors);
    }

    printf("Build time:  %.3f ms (%.3f us per insert)\n",
           ms_build, (ms_build * 1000.0f) / (float)num_items);
    printf("Lookup time: %.3f ms (%.3f us per query)\n",
           ms_lookup, (ms_lookup * 1000.0f) / (float)num_queries);

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start_build));
    CUDA_CHECK(cudaEventDestroy(stop_build));
    CUDA_CHECK(cudaEventDestroy(start_lookup));
    CUDA_CHECK(cudaEventDestroy(stop_lookup));

    CUDA_CHECK(cudaFree(d_input_keys));
    CUDA_CHECK(cudaFree(d_input_vals));
    CUDA_CHECK(cudaFree(d_query_keys));
    CUDA_CHECK(cudaFree(d_query_vals));
    CUDA_CHECK(cudaFree(d_found));
    CUDA_CHECK(cudaFree(d_table_keys));
    CUDA_CHECK(cudaFree(d_table_vals));

    free(h_keys);
    free(h_vals);
    free(h_query_keys);
    free(h_query_vals);
    free(h_found);
}

int main() {
    // Print basic GPU information for context
    print_gpu_info();

    // Deterministic seed to keep the test repeatable
    srand(0);

    // Run a moderate-size test. You can increase this to stress the table.
    int N = 1 << 17; // 131072 elements
    test_gpu_hash_table(N);

    return 0;
}
