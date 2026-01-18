#include <cstddef>
#include <stdexcept>
#include <utility>

class SinglyLinkedList {
    struct Node {
        int data;
        Node* next;
        Node(int value, Node* next_node = nullptr) : data(value), next(next_node) {}
    };

    Node* head_ = nullptr;
    Node* tail_ = nullptr;
    std::size_t size_ = 0;

public:
    SinglyLinkedList() = default;

    // Disable copy; implement move constructor and assignment
    SinglyLinkedList(const SinglyLinkedList&) = delete;
    SinglyLinkedList& operator=(const SinglyLinkedList&) = delete;

    SinglyLinkedList(SinglyLinkedList&& other) noexcept 
        : 
}