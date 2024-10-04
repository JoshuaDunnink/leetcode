class ListNode:
    def __init__(self, val=None, next=None, previous=None):
        self.val = val
        self.next = next
        self.previous = previous

    @staticmethod
    def create_listnodes(list):
        list_node = None
        for i, val in enumerate(list):
            if i == 0:
                list_node = ListNode(val=val)
            else:
                node = list_node
                while node.next is not None:
                    node = node.next
                node.next = ListNode(val=val, previous=node)
        return list_node

    @staticmethod
    def get_list_from_nodes(node):
        simple_list = []
        while node.next is not None:
            simple_list.append(node.val)
            node = node.next
        return simple_list


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_number = False
