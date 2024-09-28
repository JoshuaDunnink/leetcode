
class MyCalendarTwo:
    """731
    https://leetcode.com/problems/my-calendar-ii/description/?envType=daily-question&envId=2024-09-27
    to be completed
    """
    def __init__(self):
        # self.bookings = defaultdict(int)
        self.bookings = {0:[], 1:[]}
        

    def book(self, start: int, end: int) -> bool:
        # if not self.check(start, end):
        #     return False
        # for i in range(start, end):
        #     self.bookings[i] += 1
        # return True
        return self.try_booking(start, end, 0)

    def try_booking(self, start, end, key=0):
        possible = [True]
        for values in self.bookings[key]:
                # [new]
                #       [old]
            if values[0] > end:
                if key >= 2:
                    possible.append(True)
                else:
                    possible.append(False)
            #       [new]
            # [old]
            if start > values[1]:
                if key >= 2:
                    possible.append(True)
                else:
                    possible.append(False)
            # [new]
            #   [old]
            if start < values[0] and end > values[1]:
                if key == 0:
                    self.try_booking(start, values[0], key)
                    self.try_booking(values[0], end, key)
                if key == 1:
                    possible.append(False)
            #      [new]
            #   [old]
            if start < values[1] and end > values[1]:
                if key == 0:
                    self.try_booking(start, values[1], key)
                    self.try_booking(values[0], end, key)
                if key == 1:
                    possible.append(False)
        if all(possible):
            self.bookings[key].append((start, end))
            return True
        else:
            return False

# cal = MyCalendarTwo()
# cal.book(0, 100)
# cal.book(50, 150)