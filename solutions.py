from typing import List, Optional
from itertools import combinations, permutations, product
import re
import pprint
from nodes import ListNode
from trie import LongestPrefixTrie
import math
import heapq


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

    def findMedianSortedArrays(
        self, nums1: List[int], nums2: List[int]
    ) -> float:
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
                    (
                        not char.isnumeric()
                        and char in ["-", "+"]
                        and not number_started
                    )
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

    def survivedRobotsHealths(
        self, positions: List[int], healths: List[int], directions: str
    ) -> List[int]:
        # TODO 2751
        def walk(robot):
            if robot[2] == "R":
                robot[0] += 1
            else:
                robot[0] -= 1
            return robot

        if directions.count("R") == len(positions) or directions.count(
            "L"
        ) == len(positions):
            return healths

        robots = {}
        for i, position in enumerate(positions):
            robots.update({i: [position, healths[i], directions[i]]})

        for key, robot in robots.items():
            robots.update({key: walk(robot)})

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
        # TODO 18 combinations = too slow -> use backtracking
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

        for index in range(len(str_x) - 1, -1, -1):
            if str_x[index] != "-":
                new_x += str_x[index]

        if -(2**31) < int(new_x) < 2**31:
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
                elif fives >= 3:
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
            if small_val not in small_done:
                for i_high, high_val in enumerate(high_vals):
                    if i_small != i_high and high_val - small_val > diff:
                        diff = high_val - small_val
                if n_arrays > 10:
                    small_done.add(small_val)
        return diff

    def combinationSum2(
        self, candidates: List[int], target: int
    ) -> List[List[int]]:
        result = set()
        for r in range(1, len(candidates) + 1):
            for combo in combinations(candidates, r):
                if sum(combo) == target:
                    result.add(tuple(combo))
        return list(result)

    # def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
    #     def backtrack(start, path, target):
    #         if target == 0:
    #             result.add(tuple(path))
    #             return
    #         if target < 0:
    #             return
    #         for i in range(start, len(candidates)):
    #             if i > start and candidates[i] == candidates[i - 1]:
    #                 continue
    #             backtrack(i + 1, path + [candidates[i]], target - candidates[i])

    #     result = set()
    #     candidates.sort()
    #     backtrack(0, [], target)
    #     return list(result)

    def letterCombinations(self, digits: str) -> List[str]:
        combinations = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz",
            "0": " ",
        }
        letters = []
        if not digits:
            return []
        for digit in digits:
            letters.append(combinations.get(digit))
        return ["".join(items) for items in product(*letters)]

        # def backtrack(index, path):
        #     if index == len(digits):
        #         result.append("".join(path))
        #         return
        #     possible_letters = combinations[digits[index]]
        #     for letter in possible_letters:
        #         backtrack(index + 1, path + [letter])

        # result = []
        # backtrack(0, [])
        # return result

    def search(self, nums: List[int], target: int) -> bool:
        # TODO 81: binary search instead of simple
        return target in nums

    def search(self, nums: List[int], target: int) -> int:
        # TODO 33: binary search instead of simple
        if target not in nums:
            return -1
        else:
            return nums.index(target)

    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        clean = []
        if head is not None:
            count = {head.val: 1}
            another = False
            if head.next is not None:
                next_head = head.next
                another = True

            while another:
                if next_head.val not in count.keys():
                    count.update({next_head.val: 1})
                else:
                    count[next_head.val] += 1
                if next_head.next == None:
                    another = False
                else:
                    next_head = next_head.next

            clean = [k for k, v in count.items() if v == 1]
            clean.sort()

        # list_node = None
        # for item in clean[::-1]:
        #     list_node = ListNode(item, list_node)
        return ListNode.create_listnodes(clean)

    def solveNQueens(self, n: int) -> List[List[str]]:
        # backtracking, iterative refinement is not possible
        """51
        Input: n = 4
        Returns: Output: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
        Explanation: There exist two distinct solutions to the 4-queens puzzle as shown above
        """
        possible_boards = []
        board = []
        for _ in range(n):
            board.append(["." if x != 0 else "Q" for x in range(n)])
            pprint.pprint(board)

    def getPermutation(self, n: int, k: int) -> str:
        """60
        The set [1, 2, 3, ..., n] contains a total of n! unique permutations.
        By listing and labeling all of the permutations in order, we get the following sequence for n = 3:

            "123"
            "132"
            "213"
            "231"
            "312"
            "321"

        Given n and k, return the kth permutation sequence.
        """

        def backtrack(sequence, i):
            if i > n:
                return
            if i == n:
                pass

    def insertGreatestCommonDivisors(
        self, head: Optional[ListNode]
    ) -> Optional[ListNode]:
        # 2807
        def get_bcd(val1, val2) -> int:
            if val1 >= val2:
                big = val1
                small = val2
            else:
                big = val2
                small = val1
            for i in range(small, 0, -1):
                if big % i == 0 and small % i == 0:
                    return i

        decomposed = []
        while True:
            decomposed.append(head.val)
            if head.next != None:
                head = head.next
            else:
                break

        new_list = []
        i = 0
        if len(decomposed) == 1:
            return head
        for num1 in decomposed:
            num2 = decomposed[i + 1]
            gcd = get_bcd(num1, num2)
            new_list.extend([num1, gcd])
            i += 1
            if i + 1 >= len(decomposed):
                new_list.extend([decomposed[-1]])
                break

        list_node = None
        for item in new_list[::-1]:
            list_node = ListNode(item, list_node)
        return list_node

    def countConsistentStrings(self, allowed: str, words: List[str]) -> int:
        # 1684
        count = 0
        for word in words:
            consistant = True
            for w_char in word:
                if w_char not in allowed:
                    consistant = False
                    break
            if consistant:
                count += 1
        return count

    def xorQueries(
        self, arr: List[int], queries: List[List[int]]
    ) -> List[int]:
        # 1310
        # brute force to slow need pre-created list
        # for query in queries:
        #     subarray = arr[query[0]:query[1]+1]
        #     addative = reduce(lambda x, y: x ^ y, subarray)
        #     outcome.append(addative)
        # OR
        # for query in queries:
        #     addative = 0
        #     subarray = arr[query[0]:query[1]+1]
        #     for num in subarray:
        #         if addative == 0:
        #             addative = num
        #         else:
        #             addative = int(bin(addative),2)^int(bin(num),2)
        #     outcome.append(addative)

        outcome = []

        def compute_prefix_xor(arr):
            prefix_xor = [0] * (len(arr) + 1)
            for i in range(len(arr)):
                prefix_xor[i + 1] = prefix_xor[i] ^ arr[i]
            return prefix_xor

        prepared = compute_prefix_xor(arr)
        for left, right in queries:
            outcome.append(prepared[left] ^ prepared[right + 1])
        return outcome

    def uncommonFromSentences(self, s1: str, s2: str) -> List[str]:
        # 884
        def update_count(count: dict = {}, string: str = ""):
            for word in string.split(" "):
                if not count.get(word):
                    count.update({word: 1})
                else:
                    count[word] += 1
            return count

        count = update_count(string=s1)
        count = update_count(count, s2)

        odd = []
        for key, value in count.items():
            if value == 1:
                odd.append(key)
        return odd

    def addTwoNumbers(
        self, l1: Optional[ListNode], l2: Optional[ListNode]
    ) -> Optional[ListNode]:
        # 2
        number1 = str(l1.val)
        number2 = str(l2.val)

        while l1.next != None:
            l1 = l1.next
            number1 += str(l1.val)
        while l2.next != None:
            l2 = l2.next
            number2 += str(l2.val)

        summation = int(number1[::-1]) + int(number2[::-1])
        list_node = None
        for item in str(summation):
            list_node = ListNode(int(item), list_node)
        return list_node

    def addBinary(self, a: str, b: str) -> str:
        # 67
        return bin(int(a, 2) + int(b, 2))[2::]
        # return bin(int("0b"+a,2) + int("0b"+b,2)).replace("0b", "")

    def shortestPalindrome(self, s: str) -> str:
        # 214 -> brute force
        """
        alternative:
        compare 2 strings step by step with its reversed
          abc
        reverse == cba

        first compare
         abc
        cba

        second
          abc
        cba
        """

        def is_palindrome(s):
            if len(s) % 2 == 1:
                center = int((len(s) + 1) / 2)
                return s[0:center] == s[center - 1 : len(s)][::-1]
            else:
                center = int(len(s) / 2)
                return s[0:center] == s[center : len(s)][::-1]

        def contains_palindrome(s):
            section = ""
            for i in range(len(s) - 1, 0, -1):
                section = s[:i]
                if is_palindrome(section):
                    return i
            return 0

        if s == None:
            return ""
        elif is_palindrome(s):
            return s

        i = contains_palindrome(s)
        word = s
        for i in range(i, len(s)):
            char = s[i]
            if i != 0:
                word = (word[::-1] + char)[::-1]
                if is_palindrome(word):
                    return word

    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        # def get_count(num1, num2):
        #     count = 0
        #     for i, char in enumerate(str(num2)):
        #         if char == str(num1)[i]:
        #             count+=1
        #         else:
        #             break
        #     return count
        # cleaned_arr1 = set(arr1)
        # arr1 = list(cleaned_arr1).sort(reverse=True)
        # cleaned_arr2 = set(arr2)
        # arr2 = list(cleaned_arr2).sort(reverse=True)
        # max_count = 0
        # for num1 in cleaned_arr1:
        #     for num2 in cleaned_arr2:
        #         if len(str(num1)) >= len(str(num2)) and len(str(num1)) > max_count:
        #             count = get_count(num1, num2)
        #         elif len(str(num2)) > max_count:
        #             count = get_count(num2, num1)
        #         max_count = count if count > max_count else max_count
        # return max_count
        """build trie

        Args:
            arr1 (List[int]): _description_
            arr2 (List[int]): _description_

        Returns:
            int: _description_
        """
        longest_prefix = LongestPrefixTrie()
        for number in arr1:
            longest_prefix.insert(number)

        longest = 0
        for number in arr2:
            number_len = len(str(number))
            if number_len > longest:
                length = longest_prefix.search(number)
                longest = length if length > longest else longest

        return longest

    def removeNthFromEnd(
        self, head: Optional[ListNode], n: int
    ) -> Optional[ListNode]:
        # TODO
        node = head
        i = 1
        while node.next is not None:
            previous = node
            node = head.next
            i += 1
            if i == n:
                previous.next = node.next
                break
        return head

    def canArrange(self, arr: List[int], k: int) -> bool:
        # TODO
        options = set()
        for i, num1 in enumerate(arr):
            for i2 in range(i + 1, len(arr)):
                if (num1 + arr[i2]) % k == 0:
                    options.add((num1, arr[i2]))
        options

    def dividePlayers(self, skill: List[int]) -> int:
        # 2491
        # can be speed up by sorting and taking first and last element
        sum_skill = sum(skill)
        if len(skill) == 2:
            return skill[0] * skill[1]
        if sum_skill % (len(skill) // 2) != 0:
            return -1
        else:
            search_total = int(sum_skill / (len(skill) / 2))
            selected_pairs = []
            while len(skill) != 0:
                number = skill.pop()
                index_pair = (
                    skill.index(search_total - number)
                    if int(search_total - number) in skill
                    else None
                )
                if index_pair is None:
                    return -1
                pair = skill.pop(index_pair)
                selected_pairs.append([number, pair])
        return sum([item[0] * item[1] for item in selected_pairs])

    def mergeKLists(
        self, lists: List[Optional[ListNode]]
    ) -> Optional[ListNode]:
        # 23
        all_lists = []
        if not lists:
            return None
        if len(lists) == 1:
            return lists[0]
    
        for item in lists:
            if item:
                all_lists.extend([item.val])
                while item.next is not None:
                    item = item.next
                    all_lists.extend([item.val])

        all_lists.sort()
        list_node = None
        for i, item in enumerate(all_lists):
            if i == 0:
                list_node = ListNode(val=item)
            else:
                node = list_node
                while node.next is not None:
                    node = node.next
                node.next = ListNode(val=item)
        return list_node

    def maxWidthRamp(self, nums: List[int]) -> int:
        # brute, TLE
        max_width = 0
        for i in range(len(nums)):
            j = len(nums)-1
            while i < j:
                if nums[i] <= nums[j]:
                    width = j - i
                    max_width = width if width > max_width else max_width
                j -= 1
        return max_width

    def smallestChair(self, times: List[List[int]], targetFriend: int) -> int:
        #1942 TODO: and now with heapq?
        timetable = []
        for number, (arive, leave) in enumerate(times):
            timetable.extend([[arive, number, "a"],[leave, number, "l"]])
        timetable = sorted(timetable, key=lambda item: (item[0], item[2] != "l"))

        seats = ["x"]*len(times)
        for event in timetable:
            if event[2] == "a":
                seat_number = seats.index("x")
                seats[seat_number] = event[1]
                if event[1] == targetFriend:
                    return seat_number
            else:
                seats[seats.index(event[1])] = "x"

    def maxKelements(self, nums: List[int], k: int) -> int:
        #2530 first time using heapq
        total = 0
        heap = []
        [heapq.heappush(heap, -num)for num in nums]
        for _ in range(k):
            max_val = heapq.heappop(heap)
            heapq.heappush(heap, (math.ceil(-max_val/3))*-1)
            total-=max_val
        return total
    
    def minimumSteps(self, s: str) -> int:
        #2938
        total_counter = 0
        counter = 0

        for char in s[::-1]:
            if char == "0":
                counter += 1
            else:
                total_counter += counter
        return total_counter

    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        # 209
        # # TLE
        # window = 1
        # while window <= len(nums):
        #     arr_len = len(nums)
        #     position = 0
        #     while position+window <= arr_len:
        #         if sum(nums[position:position+window]) >= target:
        #             return window
        #         position += 1
        #     window += 1
        # return 0
        # Sliding Window
        curr_sum = 0
        lengts = []
        if sum(nums) < target:
            return 0
        left = 0
        for right in range(len(nums)):
            curr_sum += nums[right]
            while curr_sum >= target:
                lengts.append(right-left+1)
                curr_sum -= nums[left]
                left += 1
        return min(lengts)

    def findSecretWord(self, words: List[str], master: 'Master') -> None:
        #google 2023
        # TODO how to get the number of geusses down???
        def get_similars(word, words, number_correct):
            possible = []
            for word_2 in words:
                matches = sum(
                    [1 for char_1 in word for char_2 in word_2 if char_1 == char_2]
                )
                if matches >= number_correct:
                    possible.append(word_2)
            return possible
        
        while words:
            word = words.pop()
            number_correct = master.geuss(word)
            if number_correct == 6:
                break
            words = get_similars(word, words, number_correct)


    def kthFactor(self, n: int, k: int) -> int:
        #1492
        count = 0
        factors = []
        for i in range(1, n+1):
            if n%i==0:
                count += 1
                factors.append(i)
                if count == k:
                    return i
        return -1

    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        # 88
        intermediate = sorted(nums1[:m]+nums2)
        for i in range(len(nums1)):
            nums1[i] = intermediate[i]

    def removeDuplicates(self, nums: List[int]) -> int:
        # 80
        i = 2
        while i < len(nums):
            if nums[i-1] == nums[i-2] == nums[i]:
                nums.remove(nums[i])
                i -= 1
            i += 1
        return len(nums)

    def majorityElement(self, nums: List[int]) -> int:
        # 169
        from collections import defaultdict
        count = defaultdict(int)
        for num in nums:
            count[num] += 1
    
        return max(count, key=count.get)
        
    def rotate(self, nums: List[int], k: int) -> None:
        # 189
        """
        Do not return anything, modify nums in-place instead.
        """
        l_nums = len(nums)
        while k > l_nums:
            k = k - l_nums
        nums[:] = (nums+nums)[abs(k-l_nums):abs(l_nums-k)+l_nums]


    def maxProfit(self, prices: List[int]) -> int:
        # 121 changed for 122
        bought = max(prices)
        profit = 0
        for num in prices:
            if num > bought:
                profit += num-bought
                bought = num
            bought = num if num < bought else bought
        return profit
        
    def hIndex(self, citations: List[int]) -> int:
        # 274
        citations.sort(reverse=True)
        count = 0
        for i, cite in enumerate(citations, start=1):
            if cite >= i:
                count += 1
            else:
                break
        return count

    def reverseWords(self, s: str) -> str:
        #151
        return ' '.join(s.strip().split()[::-1])


print(
    Solution().candy(
          [1,3,2,2,1]
    )
)
