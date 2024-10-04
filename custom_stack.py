class CustomStack(list):
    def __init__(self, maxSize: int):
        self.max_size = maxSize
        self.stack = []

    def push(self, x: int) -> None:
        if len(self.stack) < self.max_size:
            self.stack.append(x)

    def pop(self) -> int:
        return self.stack.pop() if len(self.stack) > 0 else -1

    def increment(self, k: int, val: int) -> None:
        for i, num in enumerate(self.stack):
            if i < k:
                self.stack[i] = num + val


methods = [
    "push",
    "push",
    "pop",
    "push",
    "push",
    "push",
    "increment",
    "increment",
    "pop",
    "pop",
    "pop",
    "pop",
]
params = [[1], [2], [], [2], [3], [4], [5, 100], [2, 100], [], [], [], []]
zipped = zip(methods, params)

attempts = []
attempt = CustomStack(3)
for values in zipped:
    command = values[0]
    if len(values[1]) == 0:
        vars = ""
    elif len(values[1]) == 1:
        vars = values[1][0]
    elif len(values[1]) == 2:
        vars = f"{values[1][0]},{values[1][1]}"

    eval(f"attempts.append(attempt.{command}({vars}))")
    print()
    print(attempts)
