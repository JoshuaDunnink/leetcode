
from typing import List, Optional
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for index_a, item_a in enumerate(nums):
            for index_b, item_b in enumerate(nums):
                if index_a != index_b and item_a + item_b == target:
                    return [index_a, index_b]

    def romanToInt(self, s: str) -> int:
        values = {
            "I":1,
            "V":5,
            "X":10,
            "L":50,
            "C":100,
            "D":500,
            "M":1000,
        }
        count = 0
        previous = ""
        for index, char in enumerate(s):
            if index < len(s)-1:
                if values.get(char) >= values.get(s[index+1]):
                    if not previous:
                        count += values.get(char)
                    else:
                        count += values.get(char) - values.get(previous)
                        previous = ""
                elif values.get(char) < values.get(s[index+1]):
                    previous = char
            elif previous:
                count += values.get(char) - values.get(previous)
            else:
                count += values.get(char)
        return count

    def longestCommonPrefix(self, strs: List[str]) -> str:
        common = ""
        breaking = False
        word_1 = min(strs)
        for index, char in enumerate(word_1):
            checks = []
            for item in strs:
                if index <= len(item)-1:
                    if item[index] == char:
                        checks.append(True)
                    else:
                        checks.append(False)
                        breaking = True
                        break
            if all(checks) and not breaking:
                common+=char
            elif breaking:
                break
        return common

    def isValid(self, s: str) -> bool:
        opened = []
        closing_mapping = {")": "(", "}": "{", "]": "["}
    
        for char in s:
            if char in closing_mapping.keys():
                if not opened or opened.pop() != closing_mapping[char]:
                    return False
            else:
                opened.append(char)
        return not opened

    def removeElement(self, nums: List[int], val: int) -> int:
        new_list = []
        count = 0
        for item in nums:
            if item == val:
                count += 1
            else:
                new_list.append(item)
        return count, new_list

    def removeDuplicates(self, nums: List[int]) -> int:
        previous = int
        count = 0
        for index, current in enumerate(nums):
            if current == previous:
                nums[index] = 101
            else:
                count +=1 
            previous = current
        nums.sort()
        return count

print(Solution().removeDuplicates(nums = [0,0,1,1,1,2,2,3,3,4]))