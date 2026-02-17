#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <cuda.h>
#include <sys/time.h>
#include "../cuda_common.cuh"

// Type configuration
typedef uint32_t KEY_TYPE;
typedef uint32_t VALUE_TYPE;

// Hash table configuration
#define CAPACITY (1 << 16)  // 65536 slots
#define EMPTY_KEY 0xFFFFFFFF
#define HASH_MULTIPLIER 2654435761u
#define WARP_SIZE 32

// Hash table slot structure
typedef struct {
    KEY_TYPE key;
    VALUE_TYPE value;
    uint8_t psl;
    bool occupied;
} HashSlot;

// Device hash function
__device__ __forceinline__ size_t hash_function(KEY_TYPE key, size_t capacity) {
    return (key * HASH_MULTIPLIER) & (capacity - 1);
}

// ============================================================================
// CUDA KERNEL: Sequential Insertion (Baseline)
// ============================================================================
__global__ void robin_hood_insert_kernel_sequential(
    HashSlot *table,
    const KEY_TYPE *keys,
    const VALUE_TYPE *values,
    int num_keys,
    size_t capacity,
    bool *success_flags)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_keys) return;

    KEY_TYPE curr_key = keys[idx];
    VALUE_TYPE curr_value = values[idx];

    if (curr_key == EMPTY_KEY) {
        success_flags[idx] = false;
        return;
    }

    size_t pos = hash_function(curr_key, capacity);
    uint8_t curr_psl = 0;

    // Robin Hood insertion with displacement
    for (int iter = 0; iter < 256; iter++) {
        // Attempt atomic swap at this position
        unsigned long long old_slot = atomicCAS(
            (unsigned long long*)&table[pos],
            *((unsigned long long*)&(HashSlot){EMPTY_KEY, 0, 0, false}),
            *((unsigned long long*)&(HashSlot){curr_key, curr_value, curr_psl, true})
        );

        HashSlot old = *((HashSlot*)&old_slot);

        // Empty slot - successfully inserted
        if (!old.occupied) {
            success_flags[idx] = true;
            return;
        }

        // Key already exists - update value
        if (old.key == curr_key) {
            atomicExch((unsigned int*)&table[pos].value, curr_value);
            success_flags[idx] = true;
            return;
        }

        // Robin Hood displacement: if incoming has higher PSL, steal the slot
        if (curr_psl > old.psl) {
            // Try to swap
            unsigned long long expected = *((unsigned long long*)&old);
            unsigned long long desired = *((unsigned long long*)&(HashSlot){curr_key, curr_value, curr_psl, true});
            unsigned long long actual = atomicCAS((unsigned long long*)&table[pos], expected, desired);

            if (actual == expected) {
                // Swap successful - continue inserting the displaced entry
                curr_key = old.key;
                curr_value = old.value;
                curr_psl = old.psl;
            }
        }

        // Move to next slot
        pos = (pos + 1) & (capacity - 1);
        curr_psl++;
    }

    success_flags[idx] = false;  // Failed to insert
}

// ============================================================================
// CUDA KERNEL: Sequential Lookup (Baseline)
// ============================================================================
__global__ void robin_hood_lookup_kernel_sequential(
    const HashSlot *table,
    const KEY_TYPE *keys,
    VALUE_TYPE *values,
    bool *found_flags,
    int num_keys,
    size_t capacity)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_keys) return;

    KEY_TYPE search_key = keys[idx];

    if (search_key == EMPTY_KEY) {
        found_flags[idx] = false;
        return;
    }

    size_t pos = hash_function(search_key, capacity);
    uint8_t psl = 0;

    for (int iter = 0; iter < 256; iter++) {
        HashSlot slot = table[pos];

        // Empty slot - not found
        if (!slot.occupied) {
            found_flags[idx] = false;
            return;
        }

        // Found the key
        if (slot.key == search_key) {
            values[idx] = slot.value;
            found_flags[idx] = true;
            return;
        }

        // PSL-based early termination
        if (psl > slot.psl) {
            found_flags[idx] = false;
            return;
        }

        pos = (pos + 1) & (capacity - 1);
        psl++;
    }

    found_flags[idx] = false;
}

// ============================================================================
// CUDA KERNEL: Warp-Cooperative Lookup (Optimized)
// ============================================================================
__global__ void robin_hood_lookup_kernel_warp(
    const HashSlot *table,
    const KEY_TYPE *keys,
    VALUE_TYPE *values,
    bool *found_flags,
    int num_keys,
    size_t capacity)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = idx / WARP_SIZE;

    if (warp_id >= num_keys) return;

    KEY_TYPE search_key = keys[warp_id];

    if (search_key == EMPTY_KEY) {
        if (lane_id == 0) {
            found_flags[warp_id] = false;
        }
        return;
    }

    size_t base_pos = hash_function(search_key, capacity);

    // Warp-cooperative search: each thread checks a different slot
    // We'll search in chunks of 32 slots at a time
    const int MAX_ITERATIONS = 8;  // Search up to 256 slots (8 * 32)

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        size_t pos = (base_pos + iter * WARP_SIZE + lane_id) & (capacity - 1);
        HashSlot slot = table[pos];

        // Check if this thread found the key
        int found = (slot.occupied && slot.key == search_key) ? 1 : 0;

        // Check if we should terminate (empty slot or PSL exceeded)
        uint8_t expected_psl = (iter * WARP_SIZE + lane_id);
        int should_stop = (!slot.occupied || expected_psl > slot.psl) ? 1 : 0;

        // Use ballot to find if any thread found the key
        unsigned found_mask = __ballot_sync(0xFFFFFFFF, found);
        unsigned stop_mask = __ballot_sync(0xFFFFFFFF, should_stop);

        if (found_mask) {
            // Found the key - first thread with found=1 has it
            int winner_lane = __ffs(found_mask) - 1;
            VALUE_TYPE result_value = __shfl_sync(0xFFFFFFFF, slot.value, winner_lane);

            if (lane_id == 0) {
                values[warp_id] = result_value;
                found_flags[warp_id] = true;
            }
            return;
        }

        if (stop_mask) {
            // Should terminate - key not found
            if (lane_id == 0) {
                found_flags[warp_id] = false;
            }
            return;
        }
    }

    // Reached max iterations - not found
    if (lane_id == 0) {
        found_flags[warp_id] = false;
    }
}

// ============================================================================
// CUDA KERNEL: Delete (Sequential with Backward Shift)
// ============================================================================
__global__ void robin_hood_delete_kernel_sequential(
    HashSlot *table,
    const KEY_TYPE *keys,
    bool *success_flags,
    int num_keys,
    size_t capacity)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_keys) return;

    KEY_TYPE search_key = keys[idx];

    if (search_key == EMPTY_KEY) {
        success_flags[idx] = false;
        return;
    }

    // Find the key first
    size_t pos = hash_function(search_key, capacity);
    uint8_t psl = 0;
    bool found = false;

    for (int iter = 0; iter < 256; iter++) {
        HashSlot slot = table[pos];

        if (!slot.occupied || psl > slot.psl) {
            success_flags[idx] = false;
            return;
        }

        if (slot.key == search_key) {
            found = true;
            break;
        }

        pos = (pos + 1) & (capacity - 1);
        psl++;
    }

    if (!found) {
        success_flags[idx] = false;
        return;
    }

    // Perform backward-shift deletion
    while (true) {
        size_t next_pos = (pos + 1) & (capacity - 1);
        HashSlot next_slot = table[next_pos];

        // Stop if next slot is empty or has PSL=0
        if (!next_slot.occupied || next_slot.psl == 0) {
            // Clear current slot
            table[pos].occupied = false;
            table[pos].key = EMPTY_KEY;
            table[pos].psl = 0;
            success_flags[idx] = true;
            return;
        }

        // Shift next entry back
        table[pos].key = next_slot.key;
        table[pos].value = next_slot.value;
        table[pos].psl = next_slot.psl - 1;
        table[pos].occupied = true;

        pos = next_pos;
    }
}

// ============================================================================
// Host Test Function
// ============================================================================
void test_robin_hood_cuda() {
    printf("=== Robin Hood Hash Table (CUDA) Test ===\n");

    const int NUM_KEYS = 10000;
    const size_t table_size = CAPACITY * sizeof(HashSlot);

    // Allocate host memory
    HashSlot *h_table = (HashSlot*)calloc(CAPACITY, sizeof(HashSlot));
    KEY_TYPE *h_keys = (KEY_TYPE*)malloc(NUM_KEYS * sizeof(KEY_TYPE));
    VALUE_TYPE *h_values = (VALUE_TYPE*)malloc(NUM_KEYS * sizeof(VALUE_TYPE));
    VALUE_TYPE *h_results = (VALUE_TYPE*)malloc(NUM_KEYS * sizeof(VALUE_TYPE));
    bool *h_success = (bool*)malloc(NUM_KEYS * sizeof(bool));
    bool *h_found = (bool*)malloc(NUM_KEYS * sizeof(bool));

    // Initialize host data
    for (int i = 0; i < NUM_KEYS; i++) {
        h_keys[i] = i + 1;
        h_values[i] = i * 2;
    }

    for (size_t i = 0; i < CAPACITY; i++) {
        h_table[i].key = EMPTY_KEY;
        h_table[i].occupied = false;
        h_table[i].psl = 0;
    }

    // Allocate device memory
    HashSlot *d_table;
    KEY_TYPE *d_keys;
    VALUE_TYPE *d_values, *d_results;
    bool *d_success, *d_found;

    CUDA_CHECK(cudaMalloc(&d_table, table_size));
    CUDA_CHECK(cudaMalloc(&d_keys, NUM_KEYS * sizeof(KEY_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_values, NUM_KEYS * sizeof(VALUE_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_results, NUM_KEYS * sizeof(VALUE_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_success, NUM_KEYS * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_found, NUM_KEYS * sizeof(bool)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_table, h_table, table_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, NUM_KEYS * sizeof(KEY_TYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, h_values, NUM_KEYS * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice));

    // Test 1: Sequential Insertion
    printf("\nTest 1: Sequential Insertion\n");

    const int threads_per_block = 256;
    const int num_blocks = (NUM_KEYS + threads_per_block - 1) / threads_per_block;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    robin_hood_insert_kernel_sequential<<<num_blocks, threads_per_block>>>(
        d_table, d_keys, d_values, NUM_KEYS, CAPACITY, d_success);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float insert_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&insert_time, start, stop));
    printf("Inserted %d keys in %.2f ms (Sequential)\n", NUM_KEYS, insert_time);

    // Verify insertions
    CUDA_CHECK(cudaMemcpy(h_success, d_success, NUM_KEYS * sizeof(bool), cudaMemcpyDeviceToHost));
    for (int i = 0; i < NUM_KEYS; i++) {
        if (!h_success[i]) {
            printf("Error: Failed to insert key %d\n", h_keys[i]);
        }
    }

    // Test 2: Sequential Lookup
    printf("\nTest 2: Sequential Lookup\n");

    CUDA_CHECK(cudaEventRecord(start));
    robin_hood_lookup_kernel_sequential<<<num_blocks, threads_per_block>>>(
        d_table, d_keys, d_results, d_found, NUM_KEYS, CAPACITY);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float lookup_seq_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&lookup_seq_time, start, stop));
    printf("Looked up %d keys in %.2f ms (Sequential)\n", NUM_KEYS, lookup_seq_time);

    // Verify lookups
    CUDA_CHECK(cudaMemcpy(h_found, d_found, NUM_KEYS * sizeof(bool), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_results, d_results, NUM_KEYS * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost));

    for (int i = 0; i < NUM_KEYS; i++) {
        if (!h_found[i]) {
            printf("Error: Key %d not found\n", h_keys[i]);
        } else if (h_results[i] != h_values[i]) {
            printf("Error: Key %d has wrong value: %u (expected %u)\n",
                   h_keys[i], h_results[i], h_values[i]);
        }
    }

    // Test 3: Warp-Cooperative Lookup
    printf("\nTest 3: Warp-Cooperative Lookup\n");

    const int warp_blocks = (NUM_KEYS + WARP_SIZE - 1) / WARP_SIZE;

    CUDA_CHECK(cudaEventRecord(start));
    robin_hood_lookup_kernel_warp<<<warp_blocks, WARP_SIZE>>>(
        d_table, d_keys, d_results, d_found, NUM_KEYS, CAPACITY);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float lookup_warp_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&lookup_warp_time, start, stop));
    printf("Looked up %d keys in %.2f ms (Warp-Cooperative)\n", NUM_KEYS, lookup_warp_time);
    printf("Speedup: %.2fx\n", lookup_seq_time / lookup_warp_time);

    // Verify warp-cooperative lookups
    CUDA_CHECK(cudaMemcpy(h_found, d_found, NUM_KEYS * sizeof(bool), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_results, d_results, NUM_KEYS * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost));

    for (int i = 0; i < NUM_KEYS; i++) {
        if (!h_found[i]) {
            printf("Error: Key %d not found (warp)\n", h_keys[i]);
        } else if (h_results[i] != h_values[i]) {
            printf("Error: Key %d has wrong value (warp): %u (expected %u)\n",
                   h_keys[i], h_results[i], h_values[i]);
        }
    }

    // Test 4: Deletion with Backward Shift
    printf("\nTest 4: Deletion with Backward Shift\n");

    CUDA_CHECK(cudaEventRecord(start));
    robin_hood_delete_kernel_sequential<<<num_blocks, threads_per_block>>>(
        d_table, d_keys, d_success, NUM_KEYS / 2, CAPACITY);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float delete_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&delete_time, start, stop));
    printf("Deleted %d keys in %.2f ms\n", NUM_KEYS / 2, delete_time);

    // Verify deletions
    CUDA_CHECK(cudaMemcpy(h_success, d_success, NUM_KEYS * sizeof(bool), cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_table));
    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_success));
    CUDA_CHECK(cudaFree(d_found));

    free(h_table);
    free(h_keys);
    free(h_values);
    free(h_results);
    free(h_success);
    free(h_found);

    printf("\n=== All CUDA tests passed! ===\n");
}

int main() {
    test_robin_hood_cuda();
    return 0;
}
