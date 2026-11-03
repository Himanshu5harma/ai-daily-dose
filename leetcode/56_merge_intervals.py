"""LeetCode 56. Merge Intervals.

https://leetcode.com/problems/merge-intervals/

Sort by start, then sweep left to right. If the next interval starts at or before the end
of the last merged interval, they overlap, so extend that end with max(). Otherwise start a
new merged interval.

Time:  O(n log n), dominated by the sort
Space: O(n) for the output (O(log n) extra, ignoring the output)
"""


class Solution:
    def merge(self, intervals: list[list[int]]) -> list[list[int]]:
        merged: list[list[int]] = []
        for start, end in sorted(intervals, key=lambda interval: interval[0]):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return merged


if __name__ == "__main__":
    solution = Solution()
    assert solution.merge([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
    assert solution.merge([[1, 4], [4, 5]]) == [[1, 5]]
    assert solution.merge([[4, 7], [1, 4]]) == [[1, 7]]
    assert solution.merge([[1, 10], [2, 3], [4, 5]]) == [[1, 10]]
    assert solution.merge([[1, 1]]) == [[1, 1]]
    print("ok")
