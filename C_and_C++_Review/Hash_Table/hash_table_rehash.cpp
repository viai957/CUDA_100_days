// Implementation of Hash Table with Rehashing
#include <iostream>
#include <cstring>

// Linked List node
struct node {
    char* key;
    char* value;
    struct node* next;
};

// Constructor-like function for node
struct node* createNode(char* key, char* value, struct node* next) {
    struct node* newNode = (struct node*)malloc(sizeof(struct node));
    newNode->key = key;
    newNode->value = value;
    newNode->next = next;
    return newNode;
}

struct hashMap {
    // Current number of elements and capacity
    int numOfElements, capacity;
    // Array of linked lists
    struct node** arr;
};

// Initialize hashMap with given capacity
void initializeHashMap(struct hashMap* mp, int capacity) {
    mp->capacity = capacity;
    mp->numOfElements = 0;
    mp->arr = (struct node**)calloc(mp->capacity, sizeof(struct node*));
}

// Hash function to compute bucket index
int hashFunction(struct hashMap* mp, char* key) {
    int sum = 0, factor = 31;
    for (int i = 0; i < strlen(key); i++) {
        sum = ((sum % mp->capacity) + (((int)key[i] * factor)) % mp->capacity) % mp->capacity;
        factor = ((factor % __INT16_MAX__) * (31 % __INT16_MAX__)) % __INT16_MAX__;
    }
    return sum;
}

// Rehashing function - doubles capacity and redistributes elements
void rehash(struct hashMap* mp) {
    // Store old array and capacity
    struct node** oldArr = mp->arr;
    int oldCapacity = mp->capacity;

    // Double the capacity
    mp->capacity *= 2;
    mp->numOfElements = 0;

    // Create new array with doubled capacity
    mp->arr = (struct node**)calloc(mp->capacity, sizeof(struct node*));

    // Redistribute all elements from old array to new array
    for (int i = 0; i < oldCapacity; i++) {
        struct node* current = oldArr[i];

        // Traverse the linked list at this bucket
        while (current != NULL) {
            struct node* next = current->next;

            // Recompute hash for new capacity
            int bucketIndex = hashFunction(mp, current->key);

            // Insert at head of new bucket
            current->next = mp->arr[bucketIndex];
            mp->arr[bucketIndex] = current;
            mp->numOfElements++;

            current = next;
        }
    }

    // Free old array (not the nodes, they were moved)
    free(oldArr);

    printf("Rehashed! New capacity: %d\n", mp->capacity);
}

// Insert key-value pair with rehashing
void insert(struct hashMap* mp, char* key, char* value) {
    // Check load factor (0.75 threshold)
    double loadFactor = (double)(mp->numOfElements + 1) / mp->capacity;
    if (loadFactor > 0.75) {
        rehash(mp);
    }

    int bucketIndex = hashFunction(mp, key);
    struct node* newNode = createNode(key, value, NULL);

    // Check if key already exists (update case)
    struct node* current = mp->arr[bucketIndex];
    while (current != NULL) {
        if (strcmp(current->key, key) == 0) {
            // Key exists, update value
            current->value = value;
            free(newNode);
            return;
        }
        current = current->next;
    }

    // Key doesn't exist, insert at head
    newNode->next = mp->arr[bucketIndex];
    mp->arr[bucketIndex] = newNode;
    mp->numOfElements++;
}

// Delete key from hash table
void deleteKey(struct hashMap* mp, char* key) {
    int bucketIndex = hashFunction(mp, key);
    struct node* prevNode = NULL;
    struct node* currNode = mp->arr[bucketIndex];

    while (currNode != NULL) {
        if (strcmp(key, currNode->key) == 0) {
            // Head node deletion
            if (currNode == mp->arr[bucketIndex]) {
                mp->arr[bucketIndex] = currNode->next;
            }
            // Middle or last node deletion
            else {
                prevNode->next = currNode->next;
            }
            free(currNode);
            mp->numOfElements--;
            return;
        }
        prevNode = currNode;
        currNode = currNode->next;
    }
}

// Search for key in hash table
char* search(struct hashMap* mp, char* key) {
    int bucketIndex = hashFunction(mp, key);
    struct node* bucketHead = mp->arr[bucketIndex];

    while (bucketHead != NULL) {
        if (strcmp(bucketHead->key, key) == 0) {
            return bucketHead->value;
        }
        bucketHead = bucketHead->next;
    }

    char* errorMssg = (char*)malloc(sizeof(char) * 25);
    strcpy(errorMssg, "Oops! No data found.\n");
    return errorMssg;
}

// Display hash table statistics
void displayStats(struct hashMap* mp) {
    printf("\n--- Hash Table Statistics ---\n");
    printf("Capacity: %d\n", mp->capacity);
    printf("Elements: %d\n", mp->numOfElements);
    printf("Load Factor: %.2f\n", (double)mp->numOfElements / mp->capacity);

    // Count collisions
    int usedBuckets = 0;
    int maxChainLength = 0;

    for (int i = 0; i < mp->capacity; i++) {
        if (mp->arr[i] != NULL) {
            usedBuckets++;
            int chainLength = 0;
            struct node* current = mp->arr[i];
            while (current != NULL) {
                chainLength++;
                current = current->next;
            }
            if (chainLength > maxChainLength) {
                maxChainLength = chainLength;
            }
        }
    }

    printf("Used Buckets: %d\n", usedBuckets);
    printf("Max Chain Length: %d\n", maxChainLength);
    printf("-----------------------------\n\n");
}

// Driver code
int main() {
    // Initialize with small capacity to demonstrate rehashing
    struct hashMap* mp = (struct hashMap*)malloc(sizeof(struct hashMap));
    initializeHashMap(mp, 4);

    printf("Initial capacity: %d\n\n", mp->capacity);

    // Insert elements - will trigger rehashing
    printf("Inserting elements...\n");
    insert(mp, "Yogaholic", "Anjali");
    displayStats(mp);

    insert(mp, "pluto14", "Vartika");
    displayStats(mp);

    insert(mp, "elite_Programmer", "Manish");
    displayStats(mp);

    insert(mp, "GFG", "GeeksforGeeks");
    displayStats(mp);

    insert(mp, "decentBoy", "Mayank");
    displayStats(mp);

    // Search operations
    printf("\n--- Search Results ---\n");
    printf("elite_Programmer: %s\n", search(mp, "elite_Programmer"));
    printf("Yogaholic: %s\n", search(mp, "Yogaholic"));
    printf("pluto14: %s\n", search(mp, "pluto14"));
    printf("decentBoy: %s\n", search(mp, "decentBoy"));
    printf("GFG: %s\n", search(mp, "GFG"));
    printf("randomKey: %s\n", search(mp, "randomKey"));

    // Delete operation
    printf("\n--- After Deletion ---\n");
    deleteKey(mp, "decentBoy");
    printf("decentBoy: %s\n", search(mp, "decentBoy"));
    displayStats(mp);

    // Insert more elements to trigger another rehash
    printf("Inserting more elements...\n");
    insert(mp, "coder123", "Rahul");
    insert(mp, "techguru", "Priya");
    insert(mp, "devmaster", "Amit");
    displayStats(mp);

    return 0;
}
