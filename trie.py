from nodes import TrieNode


class LongestPrefixTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, number):
        node = self.root
        for digit in str(number):
            if digit not in node.children:
                node.children[digit] = TrieNode()
            node = node.children[digit]
        node.is_end_of_number = True

    def search(self, number):
        node = self.root
        for index, digit in enumerate(str(number)):
            if digit not in node.children:
                return index
            node = node.children[digit]
        return index + 1
