#include <stdio.h>

int main() {
    int value = 10;
    int *ptr = &value;
    int **ptr2 = &ptr;
    int ***ptr3 = &ptr2;

    printf("Value: %d\n", ***ptr3); // Output: 10 -> value of value
    printf("Address of value: %p\n", &value); // Output: 0x7ffee13799c4 -> memory address of value
    printf("Address of ptr1: %p\n", ptr); // Output: 0x7ffee13799c4 -> memory address of ptr
    printf("Address of ptr2: %p\n", ptr2); // Output: 0x7ffee13799c4 -> memory address of ptr2
    printf("Address of ptr3: %p\n", ptr3); // Output: 0x7ffee13799c4 -> memory address of ptr3
}

