// Vector Class in C++
#include <vector>
#include <iostream>
using namespace std;
template <typename T> class vectorClass {
    // arr is the integer pointer
    // which stores the address of our vector
    T* arr;

    // capacity is the total storage
    // capacity of the vector
    int capacity;

    // current is the number of elments
    // currently present in the vector
    int current;

public:
    // Default constructir to initialize
    // an initial capacity of 1 element and
    // allocating storage using dynamic allocation
    vectorClass() {
        arr = new T[1];
        capacity = 1;
        current = 0;
    }
    // destructure to dealocate storage allocated by dynamic
    // allocation to prevent memory leak
    ~vectorClass() {
        delete[] arr;
    }

    // Function to add an element at the last
    void push(T data) {
        // if the number of elements is equal to the capacity
        // that means we don't have space
        // to accomidate more elements. We need to double the capacity
        if (current == capacity) {
            T* temp = new T[2 * capacity];
            // copying old elements into new array
            for (int i = 0; i < capacity; i++){
                temp[i] = arr[i];
            }
            // deleting previous array
            delete[] arr;
            // updating new capacity
            capacity *= 2;
            // updating the pointer
            arr = temp;
        }

        // Inserting data
        arr[current] = data;
        current++;
    }

    // function to extract element at any index
    T get(int index) const {
        // if index is out of range, return default value
        if (index < 0 || index >= current) {
            return T();
        }
        return arr[index];
    }

    // function to delete last element
    void pop() { current--; }

    // function to get size of the vector
    int size() { return current; }

    // function to get capacity of the vector
    int getcapacity() { return capacity; }

    // function to print array elements
    void print(){
        for (int i = 0; i < current; i++){
            cout << arr[i] << " ";
        }
        cout << endl;
    }
};


// Driver code
int main()
{
    vectorClass<int> v;
    vectorClass<char> v1;
    v.push(10);
    v.push(20);
    v.push(30);
    v.push(40);
    v.push(50);
    v1.push(71);
    v1.push(72);
    v1.push(73);
    v1.push(74);

    cout << "Vector size : " << v.size() << endl;
    cout << "Vector capacity : " << v.getcapacity() << endl;

    cout << "Vector elements : ";
    v.print();

    v.push(100);

    cout << "\nAfter updating 1st index" << endl;

    cout << "Vector elements of type int : " << endl;
    v.print();
    // This was possible because we used templates
    cout << "Vector elements of type char : " << endl;
    v1.print();
    cout << "Element at 1st index of type int: " << v.get(1)
         << endl;
    cout << "Element at 1st index of type char: "
         << v1.get(1) << endl;

    v.pop();
    v1.pop();

    cout << "\nAfter deleting last element" << endl;

    cout << "Vector size of type int: " << v.size() << endl;
    cout << "Vector size of type char: " << v1.size()
         << endl;
    cout << "Vector capacity of type int : "
         << v.getcapacity() << endl;
    cout << "Vector capacity of type char : "
         << v1.getcapacity() << endl;

    cout << "Vector elements of type int: ";
    v.print();
    cout << "Vector elements of type char: ";
    v1.print();

    return 0;
}