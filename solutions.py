from statistics import median
from typing import List, Optional
from itertools import combinations, permutations
import re


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

    def isPalindrome(self, x: int) -> bool:
        string_val = str(x)
        check = []
        for index, char in enumerate(string_val):
            check.append(char == string_val[-(index + 1)])
        return all(check)

    def romanToInt(self, s: str) -> int:
        values = {
            "I": 1,
            "V": 5,
            "X": 10,
            "L": 50,
            "C": 100,
            "D": 500,
            "M": 1000,
        }
        count = 0
        previous = ""
        for index, char in enumerate(s):
            if index < len(s) - 1:
                if values.get(char) >= values.get(s[index + 1]):
                    if not previous:
                        count += values.get(char)
                    else:
                        count += values.get(char) - values.get(previous)
                        previous = ""
                elif values.get(char) < values.get(s[index + 1]):
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
                if index <= len(item) - 1:
                    if item[index] == char:
                        checks.append(True)
                    else:
                        checks.append(False)
                        breaking = True
                        break
            if all(checks) and not breaking:
                common += char
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
        count = 0
        for index, item in enumerate(nums):
            if item == val:
                count += 1
                nums[index] = 101
        nums.sort()
        return len(nums) - count

    def removeDuplicates(self, nums: List[int]) -> int:
        previous = int
        count = 0
        for index, current in enumerate(nums):
            if current == previous:
                nums[index] = 101
            else:
                count += 1
            previous = current
        nums.sort()
        return count

    def strStr(self, haystack: str, needle: str) -> int:
        return haystack.find(needle)

    def searchInsert(self, nums: List[int], target: int) -> int:
        for index, num in enumerate(nums):
            if num >= target:
                return index
        return len(nums)

    def lengthOfLastWord(self, s: str) -> int:
        return len(s.strip().split(" ")[-1])

    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        nums1.extend(nums2)
        nums1 = sorted(nums1)
        if len(nums1) % 2 == 1:
            middle = int((len(nums1) + 1) / 2) - 1
            return nums1[middle]
        else:
            middle = int(len(nums1) / 2)
            middle_1 = middle - 1
            return (nums1[middle] + nums1[middle_1]) / 2
        # return median(sorted(nums1+nums2))

    def myAtoi(self, s: str) -> int:
        number = ""
        number_started = False
        try:
            number = int(s.strip())
        except ValueError:
            for char in s.strip():
                if (
                    (not char.isnumeric() and char in ["-", "+"] and not number_started)
                    or (char.isnumeric() and not number_started)
                    or (char.isnumeric() and number_started)
                ):
                    number += char
                    number_started = True
                elif (not char.isnumeric() and not number_started) or (
                    not char.isnumeric() and number_started
                ):
                    break
        try:
            number = int(number)
            if int(number) <= -2147483648:
                return -2147483648
            elif int(number) >= 2147483648:
                return 2147483647
            return int(number)
        except ValueError:
            return 0

    def isMatch(self, s: str, p: str) -> bool:
        while "**" in p:
            p = p.replace("**", "*")
        return re.fullmatch(rf"{p}", s) == True

    def maximumGain(self, s: str, x: int, y: int) -> int:
        def get_score(score: int, points: int, stack: list, string: str):
            new_stack = []
            for char in stack:
                if (
                    len(new_stack) >= 1
                    and new_stack[-1] == string[0]
                    and char == string[1]
                ):
                    score += points
                    new_stack.pop()
                else:
                    new_stack.append(char)
            return score, new_stack

        score = 0
        if x > y:
            high = "ab"
            high_score = x
            low = "ba"
            low_score = y
        else:
            high = "ba"
            high_score = y
            low = "ab"
            low_score = x

        stack = list(s)
        score, stack = get_score(score, high_score, stack, high)
        score, _ = get_score(score, low_score, stack, low)

        return score

    def survivedRobotsHealths(self, positions: List[int], healths: List[int], directions: str) -> List[int]:
        #TODO 2751
        def walk(robot):
            if robot[2] == "R":
                robot[0] += 1
            else:
                robot[0] -= 1
            return robot

        if directions.count("R") == len(positions) or directions.count("L") == len(positions):
            return healths

        robots = {}
        for i, position in enumerate(positions):
            robots.update({i:[position, healths[i], directions[i]]})

        for key, robot in robots.items():
            robots.update({key:walk(robot)})
        
        for key, robot in robots.items():
            for key_2, robot_2 in robots.items():
                if not key == key_2:
                    if robot[0] == robot_2[0]:
                        if robot:
                            pass
        # scenarios:
        # after collisions they all walk the same way
        # after collisions they walk away from eachother

    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        #TODO 18 combinations = too slow
        possible = []
        for combination in combinations(nums, 4):
            if sum(combination) == target:
                possible.append(tuple(sorted(combination)))
        cleanup = [list(item) for item in set(possible)]
        return cleanup

    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        result = []
        for mutation in permutations(nums):
            result.append(mutation)
        return [list(num) for num in set(result)]
    
    def reverse(self, x: int) -> int:
        new_x = str()
        str_x = str(x)
        if str_x[0] == "-":
            new_x += "-"
        
        for index in range(len(str_x)-1, -1 , -1):
            if str_x[index] != "-":
                new_x += str_x[index]

        if -2**31 < int(new_x) < 2**31:
            return int(new_x)
        else:
            return 0

    def lemonadeChange(self, bills: List[int]) -> bool:
        fives = 0
        tens = 0
        for bill in bills:
            if bill == 5:
                fives += 1
            elif bill == 10:
                if fives >= 1:
                    tens += 1
                    fives -= 1
                else:
                    return False
            elif bill == 20:
                if tens >= 1 and fives >= 1:
                    fives -= 1
                    tens -= 1
                elif fives>= 3:
                    fives -= 3
                else:
                    return False
        return True

    def maxDistance(self, arrays: List[List[int]]) -> int:
        diff = 0
        small_vals = [items[0] for items in arrays]
        high_vals = [items[-1] for items in arrays]
        small_done = set()
        n_arrays = len(arrays)
        for i_small, small_val in enumerate(small_vals):
            if  small_val not in small_done:
                for i_high, high_val in enumerate(high_vals):
                    if i_small != i_high and high_val - small_val > diff:
                        diff = high_val - small_val
                if n_arrays > 10:
                    small_done.add(small_val)
        return diff
    
    # def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
    #     result = []
    #     for r in range(1, len(candidates) + 1):
    #         for combo in combinations(candidates, r):
    #             if sum(combo) == target:
    #                 result.append(combo)
    #     return result

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        def backtrack(start, path, target):
            if target == 0:
                result.add(tuple(path))
                return
            if target < 0:
                return
            for i in range(start, len(candidates)):
                if i > start and candidates[i] == candidates[i - 1]:
                    continue
                backtrack(i + 1, path + [candidates[i]], target - candidates[i])

        result = set()
        candidates.sort()
        backtrack(0, [], target)
        return list(result)


print(
    Solution().combinationSum2([10,1,2,7,6,1,5], 8)
)

