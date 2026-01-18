#include <stdio.h>

int main() {
    int num = 10;
    float fnum = 3.14;
    void *vptr;

    vptr = &num;
    printf("Integer: %d\n", *(int*)vptr); // Output: 10 -> value of num
    // vptr is a memory address "&num" but is stored as a void pointer (no data type)
    // We can't dereference a void pointer, so we cast it to an integer pointer to store the integer
    // Then we dereference the integer pointer to get the value of num

    vptr = &fnum;
    printf("Float: %.2f\n", *(float*)vptr); // Output: 3.14 -> value of fnum
    // vptr is a memory address "&fnum" but is stored as a void pointer (no data type)
    // We can't dereference a void pointer, so we cast it to a float pointer to store the float
    // Then we dereference the float pointer to get the value of fnum
}

// void pointers are useful when you don't know the data type of the pointer
// but you need to store the pointer in a variable
// fun fact: malloc returns a void pointer but we can see it as a pointer to specific data type after the cast