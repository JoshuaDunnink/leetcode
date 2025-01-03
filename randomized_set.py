from random import choice


class RandomizedSet:
    #380
    def __init__(self):
        self.items = set()

    def insert(self, val: int) -> bool:
        if val not in self.items:
            self.items.add(val)
            return True
        return False

    def remove(self, val: int) -> bool:
        if val in self.items:
            self.items.remove(val)
            return True
        return False

    def getRandom(self) -> int:
        return choice(list(self.items))