"""LeetCode 3665. Twisted Mirror Path Count.

https://leetcode.com/problems/twisted-mirror-path-count/

DP over cells, split by the direction the robot is moving when it tries to enter a cell:
`right[i][j]` and `down[i][j]` count the ways to arrive at (i, j) moving right / down.

- Empty cell: the robot stops, so the total continues both right and down.
- Mirror cell: arriving right becomes a move down into (i + 1, j); arriving down becomes
  a move right into (i, j + 1). Chains of mirrors need no loop, because each mirror just
  passes its counts on to the next cell.

Every transfer goes to a later cell in row-major order, so one pass is enough.
Moves that leave the grid are dropped.

Time:  O(m * n)
Space: O(m * n)

`SolutionOptimized` keeps the same transitions with O(n) space: `down[j]` holds arrivals
into the current row from above, and a single `right` value carries arrivals from the left.

Time:  O(m * n)
Space: O(n)
"""

MOD = 10**9 + 7


class Solution:
    def uniquePaths(self, grid: list[list[int]]) -> int:
        m, n = len(grid), len(grid[0])
        right = [[0] * n for _ in range(m)]
        down = [[0] * n for _ in range(m)]
        right[0][0] = 1  # the start is an empty cell; treat it as the first arrival

        for i in range(m):
            for j in range(n):
                r, d = right[i][j], down[i][j]
                if grid[i][j] == 0:
                    total = (r + d) % MOD
                    if i == m - 1 and j == n - 1:
                        return total
                    if j + 1 < n:
                        right[i][j + 1] = (right[i][j + 1] + total) % MOD
                    if i + 1 < m:
                        down[i + 1][j] = (down[i + 1][j] + total) % MOD
                else:
                    if i + 1 < m:
                        down[i + 1][j] = (down[i + 1][j] + r) % MOD
                    if j + 1 < n:
                        right[i][j + 1] = (right[i][j + 1] + d) % MOD
        return 0


class SolutionOptimized:
    def uniquePaths(self, grid: list[list[int]]) -> int:
        m, n = len(grid), len(grid[0])
        down = [0] * n
        for i in range(m):
            right = 1 if i == 0 else 0
            for j in range(n):
                d = down[j]
                if grid[i][j] == 0:
                    total = (right + d) % MOD
                    down[j] = right = total
                else:
                    down[j], right = right, d
        return down[n - 1]


if __name__ == "__main__":
    examples = [
        ([[0, 1, 0], [0, 0, 1], [1, 0, 0]], 5),
        ([[0, 0], [0, 0]], 2),
        ([[0, 1, 1], [1, 1, 0]], 1),
    ]
    for solution in (Solution(), SolutionOptimized()):
        for grid, expected in examples:
            assert solution.uniquePaths(grid) == expected
    print("ok")
