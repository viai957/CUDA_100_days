#include <stdio.h> // Standars Input/Output header file 

// & "address of" operator
// * "dereference" operator

int main() {
    int x = 10;
    int* ptr = &x; // & is used ti get the memeory address of the variable x
    printf("Address of x: %p\n", ptr); // Output: 0x7ffee13799c4 -> memory address of x
    Printf("Value of x: %d\n", *ptr); // Output: 10 -> value of x
    Printf("Value of x: %d\n", x); 
}