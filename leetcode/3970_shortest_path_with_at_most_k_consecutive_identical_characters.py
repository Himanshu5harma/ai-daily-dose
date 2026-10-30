"""LeetCode 3970. Shortest Path With At Most K Consecutive Identical Characters.

https://leetcode.com/problems/shortest-path-with-at-most-k-consecutive-identical-characters/

Dijkstra over states (node, run), where `run` is the length of the run of identical
labels the path currently ends with (1..k). Moving u -> v extends the run when the
labels match and resets it to 1 otherwise; moves that would exceed k are skipped.

Time:  O(E * k * log(E * k))
Space: O(n * k)
"""

import heapq


class Solution:
    def shortestPath(self, n: int, edges: list[list[int]], labels: str, k: int) -> int:
        adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        for u, v, w in edges:
            adj[u].append((v, w))

        inf = float("inf")
        dist = [[inf] * (k + 1) for _ in range(n)]  # dist[node][run]
        dist[0][1] = 0
        heap = [(0, 0, 1)]  # (cost, node, run)

        while heap:
            cost, u, run = heapq.heappop(heap)
            if cost > dist[u][run]:
                continue
            if u == n - 1:
                return cost
            for v, w in adj[u]:
                new_run = run + 1 if labels[v] == labels[u] else 1
                if new_run > k:
                    continue
                new_cost = cost + w
                if new_cost < dist[v][new_run]:
                    dist[v][new_run] = new_cost
                    heapq.heappush(heap, (new_cost, v, new_run))
        return -1


if __name__ == "__main__":
    solution = Solution()
    edges = [[0, 1, 1], [1, 2, 1], [0, 2, 3]]
    assert solution.shortestPath(3, edges, "aab", 1) == 3
    assert solution.shortestPath(3, edges, "aab", 2) == 2
    assert solution.shortestPath(3, [[0, 1, 1], [1, 2, 1]], "aaa", 2) == -1
    assert solution.shortestPath(1, [], "a", 1) == 0
    print("ok")
