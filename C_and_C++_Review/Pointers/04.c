#include <stdio.h>
#include <stdlib.h>

int main() {
    // Initializing a pointer to NULL
    int* ptr = NULL;
    printf("1. Initializing ptr to NULL: %p\n", ptr); // Output: 0x0 -> NULL pointer

    // Check for NULL pointer
    if (ptr == NULL){
        printf("2. ptr is NULL, cannot dereference\n");
    }

    // Allocate memory to the pointer
    ptr = (int*)malloc(sizeof(int));
    if (ptr == NULL){
        printf("3. Memory allocation failed\n");
        return 1;
    }

    printf("4. After allocation, ptr value: %p\n", ptr); // Output: 0x7ffee13799c4 -> memory address of the allocated memory

    // Safe to use ptr after NULL check and allocation
    *ptr = 10;
    printf("5. Value at ptr: %d\n", *ptr); // Output: 10 -> value at the memory address

    // Clean up
    free(ptr);
    ptr = NULL; // Set to NULL to avoid dangling pointer
    printf("6. After free, ptr value: %p\n", ptr); // Output: 0x0 -> NULL pointer
    // Demonstrate safety of NULL pointer
    if (ptr == NULL){
        printf("7. ptr is NULL, safely avoided use after free\n");
    }

    return 0;
}