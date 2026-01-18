#include<stdio.h>

int main() {
    int arr[] = {12, 13, 14, 15};
    int *ptr = arr; // ptr points to the first element of the array (default in C)

    printf("Position one: %d\n", *ptr); // Output: 12 -> value at the memory address

    for (int i = 0; i < 4; i++){
        printf("%d\t", *ptr);
        printf("%p\t", ptr);
        printf("\n");
        ptr++;
    }
    // Output:
    // Position one: 12
    // 12 0x7ffee13799c4 13 0x7ffee13799c8 14 0x7ffee13799cc 15 0x7ffee13799d0
}

// NOTE : The pointer is incremented by the size of int = 4 bytes * 8 bits/bytes = 32 bits = 4 bytes