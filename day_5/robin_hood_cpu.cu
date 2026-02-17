#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>

// Type configuration - easy to swap for different key/value types
typedef uint32_t KEY_TYPE;
typedef uint32_t VALUE_TYPE;

// Hash table configuration
#define CAPACITY (1 << 16)  // 65536 slots (power of 2 for fast modulo)
#define EMPTY_KEY 0xFFFFFFFF  // Special value to indicate empty slot

// Multiplicative hashing with Fibonacci golden ratio
#define HASH_MULTIPLIER 2654435761u

// Hash table slot structure
typedef struct {
    KEY_TYPE key;
    VALUE_TYPE value;
    uint8_t psl;      // Probe Sequence Length - distance from home bucket
    bool occupied;
} HashSlot;

// Hash table structure
typedef struct {
    HashSlot *slots;
    size_t capacity;
    size_t size;      // Number of occupied slots
} RobinHoodHashTable;

// Hash function - multiplicative hashing
static inline size_t hash_function(KEY_TYPE key, size_t capacity) {
    // Use bit mask for power-of-2 capacity (faster than modulo)
    return (key * HASH_MULTIPLIER) & (capacity - 1);
}

// Calculate PSL for a given key at a given position
static inline uint8_t calculate_psl(KEY_TYPE key, size_t position, size_t capacity) {
    size_t home = hash_function(key, capacity);
    if (position >= home) {
        return (uint8_t)(position - home);
    } else {
        // Wrapped around
        return (uint8_t)(capacity - home + position);
    }
}

// Initialize hash table
RobinHoodHashTable* robin_hood_create(size_t capacity) {
    RobinHoodHashTable *table = (RobinHoodHashTable*)malloc(sizeof(RobinHoodHashTable));
    table->capacity = capacity;
    table->size = 0;
    table->slots = (HashSlot*)calloc(capacity, sizeof(HashSlot));

    // Initialize all slots as empty
    for (size_t i = 0; i < capacity; i++) {
        table->slots[i].key = EMPTY_KEY;
        table->slots[i].occupied = false;
        table->slots[i].psl = 0;
    }

    return table;
}

// Destroy hash table
void robin_hood_destroy(RobinHoodHashTable *table) {
    if (table) {
        free(table->slots);
        free(table);
    }
}

// Insert key-value pair into hash table
bool robin_hood_insert(RobinHoodHashTable *table, KEY_TYPE key, VALUE_TYPE value) {
    if (key == EMPTY_KEY) {
        fprintf(stderr, "Error: Cannot insert EMPTY_KEY\n");
        return false;
    }

    if (table->size >= table->capacity * 0.9) {
        fprintf(stderr, "Error: Hash table is too full (load factor > 0.9)\n");
        return false;
    }

    size_t pos = hash_function(key, table->capacity);
    KEY_TYPE curr_key = key;
    VALUE_TYPE curr_value = value;
    uint8_t curr_psl = 0;

    // Robin Hood insertion with displacement
    while (true) {
        HashSlot *slot = &table->slots[pos];

        // Empty slot - insert here
        if (!slot->occupied) {
            slot->key = curr_key;
            slot->value = curr_value;
            slot->psl = curr_psl;
            slot->occupied = true;
            table->size++;
            return true;
        }

        // Key already exists - update value
        if (slot->key == curr_key) {
            slot->value = curr_value;
            return true;
        }

        // Robin Hood displacement: if incoming entry is "poorer" (higher PSL),
        // steal this slot and continue inserting the displaced entry
        if (curr_psl > slot->psl) {
            // Swap: steal the slot
            KEY_TYPE temp_key = slot->key;
            VALUE_TYPE temp_value = slot->value;
            uint8_t temp_psl = slot->psl;

            slot->key = curr_key;
            slot->value = curr_value;
            slot->psl = curr_psl;

            curr_key = temp_key;
            curr_value = temp_value;
            curr_psl = temp_psl;
        }

        // Move to next slot
        pos = (pos + 1) & (table->capacity - 1);
        curr_psl++;

        // Safety check - prevent infinite loops
        if (curr_psl > 255) {
            fprintf(stderr, "Error: PSL overflow - table is too full\n");
            return false;
        }
    }
}

// Lookup value for a given key
bool robin_hood_lookup(RobinHoodHashTable *table, KEY_TYPE key, VALUE_TYPE *value) {
    if (key == EMPTY_KEY) {
        return false;
    }

    size_t pos = hash_function(key, table->capacity);
    uint8_t psl = 0;

    while (true) {
        HashSlot *slot = &table->slots[pos];

        // Empty slot - key not found
        if (!slot->occupied) {
            return false;
        }

        // Found the key
        if (slot->key == key) {
            if (value) {
                *value = slot->value;
            }
            return true;
        }

        // PSL-based early termination: if we've probed farther than the
        // current slot's PSL, the key cannot exist in the table
        // (it would have displaced this entry if it existed)
        if (psl > slot->psl) {
            return false;
        }

        // Continue probing
        pos = (pos + 1) & (table->capacity - 1);
        psl++;

        // Safety check
        if (psl > 255) {
            return false;
        }
    }
}

// Delete key from hash table using backward-shift deletion
bool robin_hood_delete(RobinHoodHashTable *table, KEY_TYPE key) {
    if (key == EMPTY_KEY) {
        return false;
    }

    size_t pos = hash_function(key, table->capacity);
    uint8_t psl = 0;

    // First, find the key
    while (true) {
        HashSlot *slot = &table->slots[pos];

        if (!slot->occupied) {
            return false;  // Key not found
        }

        if (slot->key == key) {
            // Found the key - now perform backward-shift deletion
            break;
        }

        if (psl > slot->psl) {
            return false;  // Key not found (PSL early termination)
        }

        pos = (pos + 1) & (table->capacity - 1);
        psl++;
    }

    // Backward-shift deletion: shift subsequent entries back
    // until we hit an empty slot or an entry with PSL=0
    while (true) {
        size_t next_pos = (pos + 1) & (table->capacity - 1);
        HashSlot *next_slot = &table->slots[next_pos];

        // Stop if next slot is empty or has PSL=0 (home position)
        if (!next_slot->occupied || next_slot->psl == 0) {
            table->slots[pos].occupied = false;
            table->slots[pos].key = EMPTY_KEY;
            table->slots[pos].psl = 0;
            table->size--;
            return true;
        }

        // Shift the next entry back
        table->slots[pos].key = next_slot->key;
        table->slots[pos].value = next_slot->value;
        table->slots[pos].psl = next_slot->psl - 1;
        table->slots[pos].occupied = true;

        pos = next_pos;
    }
}

// Test function
void test_robin_hood_cpu() {
    printf("=== Robin Hood Hash Table (CPU) Test ===\n");

    RobinHoodHashTable *table = robin_hood_create(CAPACITY);

    // Test 1: Basic insertion and lookup
    printf("\nTest 1: Basic insertion and lookup\n");
    const int NUM_KEYS = 10000;

    struct timeval start, end;
    gettimeofday(&start, NULL);

    for (int i = 0; i < NUM_KEYS; i++) {
        KEY_TYPE key = i + 1;  // Avoid EMPTY_KEY (0xFFFFFFFF)
        VALUE_TYPE value = i * 2;
        assert(robin_hood_insert(table, key, value));
    }

    gettimeofday(&end, NULL);
    float insert_time = (end.tv_sec - start.tv_sec) * 1000.0 +
                        (end.tv_usec - start.tv_usec) / 1000.0;
    printf("Inserted %d keys in %.2f ms\n", NUM_KEYS, insert_time);
    printf("Load factor: %.2f%%\n", (table->size * 100.0) / table->capacity);

    // Verify lookups
    gettimeofday(&start, NULL);

    for (int i = 0; i < NUM_KEYS; i++) {
        KEY_TYPE key = i + 1;
        VALUE_TYPE expected_value = i * 2;
        VALUE_TYPE actual_value;

        assert(robin_hood_lookup(table, key, &actual_value));
        assert(actual_value == expected_value);
    }

    gettimeofday(&end, NULL);
    float lookup_time = (end.tv_sec - start.tv_sec) * 1000.0 +
                        (end.tv_usec - start.tv_usec) / 1000.0;
    printf("Looked up %d keys in %.2f ms\n", NUM_KEYS, lookup_time);

    // Test 2: Deletion with backward shift
    printf("\nTest 2: Deletion with backward shift\n");

    gettimeofday(&start, NULL);

    for (int i = 0; i < NUM_KEYS / 2; i++) {
        KEY_TYPE key = i + 1;
        assert(robin_hood_delete(table, key));
    }

    gettimeofday(&end, NULL);
    float delete_time = (end.tv_sec - start.tv_sec) * 1000.0 +
                        (end.tv_usec - start.tv_usec) / 1000.0;
    printf("Deleted %d keys in %.2f ms\n", NUM_KEYS / 2, delete_time);

    // Verify deleted keys are gone
    for (int i = 0; i < NUM_KEYS / 2; i++) {
        KEY_TYPE key = i + 1;
        VALUE_TYPE value;
        assert(!robin_hood_lookup(table, key, &value));
    }

    // Verify remaining keys still exist
    for (int i = NUM_KEYS / 2; i < NUM_KEYS; i++) {
        KEY_TYPE key = i + 1;
        VALUE_TYPE expected_value = i * 2;
        VALUE_TYPE actual_value;

        assert(robin_hood_lookup(table, key, &actual_value));
        assert(actual_value == expected_value);
    }

    printf("Deletion test passed - backward shift working correctly\n");

    // Test 3: PSL statistics
    printf("\nTest 3: PSL statistics (measuring displacement)\n");

    robin_hood_destroy(table);
    table = robin_hood_create(CAPACITY);

    // Insert keys in a pattern that creates collisions
    for (int i = 0; i < NUM_KEYS; i++) {
        KEY_TYPE key = i + 1;
        VALUE_TYPE value = i * 2;
        robin_hood_insert(table, key, value);
    }

    // Calculate PSL statistics
    uint64_t total_psl = 0;
    uint8_t max_psl = 0;

    for (size_t i = 0; i < table->capacity; i++) {
        if (table->slots[i].occupied) {
            total_psl += table->slots[i].psl;
            if (table->slots[i].psl > max_psl) {
                max_psl = table->slots[i].psl;
            }
        }
    }

    float avg_psl = (float)total_psl / table->size;
    printf("Average PSL: %.2f\n", avg_psl);
    printf("Maximum PSL: %d\n", max_psl);
    printf("(Lower PSL values indicate better performance)\n");

    robin_hood_destroy(table);
    printf("\n=== All CPU tests passed! ===\n");
}

int main() {
    srand(42);  // Deterministic seed for reproducibility

    test_robin_hood_cpu();

    return 0;
}
