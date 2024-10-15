class MaxHeap:
    def __init__(self):
        self.heap = []

    def _left_child(self, index):
        """
        Returns the index of the left child of the given node.
        The left child of a node at index `i` is located at index
        `2*i + 1`.

        Args:
            index (int): The index of the node.

        Returns:
            int: The index of the left child.
        """
        return 2 * index + 1

    def _right_child(self, index):
        """
        Returns the index of the right child of the given node.
        The right child of a node at index `i` is located at index
        `2*i + 2`.

        Args:
            index (int): The index of the node.

        Returns:
            int: The index of the right child.
        """
        return 2 * index + 2

    def _parent(self, index):
        """
        Returns the index of the parent of the given node.
        The parent of a node at index `i` is located at index `(i - 1) // 2`.

        Args:
            index (int): The index of the node.

        Returns:
            int: The index of the parent.
        """
        return (index - 1) // 2

    def _swap(self, index1, index2):
        """
        Swaps the elements at the given indices.
        This method exchanges the values at `index1` and `index2` in the
        heap list.

        Args:
            index1 (int): The index of the first element.
            index2 (int): The index of the second element.
        """
        self.heap[index1], self.heap[index2] = self.heap[index2], self.heap[index1]

    def insert(self, value):
        """
        Inserts a new value into the heap.
        This method adds the new value to the end of the heap list and
        then moves it up to its correct position to maintain the
        max-heap property.

        Args:
            value (int): The value to be inserted.
        """
        self.heap.append(value)
        current = len(self.heap) - 1

        while current > 0 and self.heap[current] > self.heap[self._parent(current)]:
            self._swap(current, self._parent(current))
            current = self._parent(current)

    def _sink_down(self, index):
        """
        Sinks down the element at the given index to maintain the heap
        property. This method moves the element at `index` down to its
        correct position by repeatedly swapping it with its largest
        child until the max-heap property is restored.

        Args:
            index (int): The index of the element to sink down.
        """
        max_index = index
        while True:
            left_index = self._left_child(index)
            right_index = self._right_child(index)

            if (left_index < len(self.heap) and 
                    self.heap[left_index] > self.heap[max_index]):
                max_index = left_index

            if (right_index < len(self.heap) and 
                    self.heap[right_index] > self.heap[max_index]):
                max_index = right_index

            if max_index != index:
                self._swap(index, max_index)
                index = max_index
            else:
                return

    def pop(self):
        """
        Removes and returns the maximum value from the heap.
        This method removes the root of the heap (the maximum value),
        replaces it with the last element,
        and then sinks down the new root to restore the max-heap
        property.

        Returns:
            int: The maximum value in the heap, or None if the heap is empty.
        """
        if len(self.heap) == 0:
            return None

        if len(self.heap) == 1:
            return self.heap.pop()

        max_value = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._sink_down(0)

        return max_value
