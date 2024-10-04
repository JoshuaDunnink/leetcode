from solutions import ListNode


class MyCircularDeque:
    """641

    Returns:
        _type_: _description_
    """

    def __init__(self, k: int):
        self.k = k
        self.deque = None
        self.size = 0

    def insertFront(self, value: int) -> bool:
        if not self.isFull():
            if self.size < self.k:
                self.deque = ListNode(val=value, next=self.deque)
                if self.size > 0:
                    next = self.deque.next
                    next.previous = self.deque
                self.size += 1
                return True
        else:
            return False

    def insertLast(self, value: int) -> bool:
        if not self.isFull():
            if self.size == 0:
                self.deque = ListNode(val=value)
                self.size += 1
                return True
            if self.size < self.k:
                node = self.deque
                while node.next != None:
                    node = node.next
                node.next = ListNode(val=value, previous=node)
                self.size += 1
                return True
        else:
            return False

    def deleteFront(self) -> bool:
        if self.size > 0:
            self.deque = self.deque.next
            if self.size > 1:
                self.deque.previous = None
            self.size -= 1
            return True
        else:
            return False

    def deleteLast(self) -> bool:
        if self.size > 1:
            node = self.deque
            while node.next != None:
                node = node.next
                if node.next == None:
                    node = node.previous
                    node.next = None
            self.size -= 1
            return True
        if self.size == 1:
            self.deque = None
            self.size -= 1
            return True
        return False

    def getFront(self) -> int:
        return self.deque.val if self.size > 0 else -1

    def getRear(self) -> int:
        if self.size > 0:
            node = self.deque
            while node.next != None:
                node = node.next
            return node.val
        return -1

    def isFull(self) -> bool:
        return self.size == self.k

    def isEmpty(self) -> bool:
        return self.size == 0


methods = [
    "insertFront",
    "getRear",
    "deleteLast",
    "getRear",
    "insertFront",
    "insertFront",
    "insertFront",
    "insertFront",
    "isFull",
    "insertFront",
    "isFull",
    "getRear",
    "deleteLast",
    "getFront",
    "getFront",
    "insertLast",
    "deleteFront",
    "getFront",
    "insertLast",
    "getRear",
    "insertLast",
    "getRear",
    "getFront",
    "getFront",
    "getFront",
    "getRear",
    "getRear",
    "insertFront",
    "getFront",
    "getFront",
    "getFront",
    "getFront",
    "deleteFront",
    "insertFront",
    "getFront",
    "deleteLast",
    "insertLast",
    "insertLast",
    "getRear",
    "getRear",
    "getRear",
    "isEmpty",
    "insertFront",
    "deleteLast",
    "getFront",
    "deleteLast",
    "getRear",
    "getFront",
    "isFull",
    "isFull",
    "deleteFront",
    "getFront",
    "deleteLast",
    "getRear",
    "insertFront",
    "getFront",
    "insertFront",
    "insertFront",
    "getRear",
    "isFull",
    "getFront",
    "getFront",
    "insertFront",
    "insertLast",
    "getRear",
    "getRear",
    "deleteLast",
    "insertFront",
    "getRear",
    "insertLast",
    "getFront",
    "getFront",
    "getFront",
    "getRear",
    "insertFront",
    "isEmpty",
    "getFront",
    "getFront",
    "insertFront",
    "deleteFront",
    "insertFront",
    "deleteLast",
    "getFront",
    "getRear",
    "getFront",
    "insertFront",
    "getFront",
    "deleteFront",
    "insertFront",
    "isEmpty",
    "getRear",
    "getRear",
    "getRear",
    "getRear",
    "deleteFront",
    "getRear",
    "isEmpty",
    "deleteFront",
    "insertFront",
    "insertLast",
    "deleteLast",
]
params = [
    [89],
    [],
    [],
    [],
    [19],
    [23],
    [23],
    [82],
    [],
    [45],
    [],
    [],
    [],
    [],
    [],
    [74],
    [],
    [],
    [98],
    [],
    [99],
    [],
    [],
    [],
    [],
    [],
    [],
    [8],
    [],
    [],
    [],
    [],
    [],
    [75],
    [],
    [],
    [35],
    [59],
    [],
    [],
    [],
    [],
    [22],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [21],
    [],
    [26],
    [63],
    [],
    [],
    [],
    [],
    [87],
    [76],
    [],
    [],
    [],
    [26],
    [],
    [67],
    [],
    [],
    [],
    [],
    [36],
    [],
    [],
    [],
    [72],
    [],
    [87],
    [],
    [],
    [],
    [],
    [85],
    [],
    [],
    [91],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [34],
    [44],
    [],
]
zipped = zip(methods, params)

attempts = []
attempt = MyCircularDeque(77)
for values in zipped:
    command = values[0]
    vars = values[1][0] if len(values[1]) > 0 else ""
    eval(f"attempts.append(attempt.{command}({vars}))")
    print(attempts)
