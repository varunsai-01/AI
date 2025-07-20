#BREATH FIRST SEARCH

from collections import deque

def min_steps(m, n, d):
    if d > max(m, n):
        return -1

    q = deque([(0, 0, 0)])  # (jug1, jug2, steps)
    visited = [[False] * (n + 1) for _ in range(m + 1)]
    visited[0][0] = True

    while q:
        jug1, jug2, steps = q.popleft()

        if jug1 == d or jug2 == d:
            return steps

        # Fill jug2
        if not visited[jug1][n]:
            visited[jug1][n] = True
            q.append((jug1, n, steps + 1))

        # Fill jug1
        if not visited[m][jug2]:
            visited[m][jug2] = True
            q.append((m, jug2, steps + 1))

        # Empty jug1
        if not visited[0][jug2]:
            visited[0][jug2] = True
            q.append((0, jug2, steps + 1))

        # Empty jug2
        if not visited[jug1][0]:
            visited[jug1][0] = True
            q.append((jug1, 0, steps + 1))

        # Pour jug1 -> jug2
        pour1to2 = min(jug1, n - jug2)
        if not visited[jug1 - pour1to2][jug2 + pour1to2]:
            visited[jug1 - pour1to2][jug2 + pour1to2] = True
            q.append((jug1 - pour1to2, jug2 + pour1to2, steps + 1))

        # Pour jug2 -> jug1
        pour2to1 = min(jug2, m - jug1)
        if not visited[jug1 + pour2to1][jug2 - pour2to1]:
            visited[jug1 + pour2to1][jug2 - pour2to1] = True
            q.append((jug1 + pour2to1, jug2 - pour2to1, steps + 1))

    return -1


if __name__ == "__main__":
    m, n, d = 4, 3, 2
    result = min_steps(m, n, d)
    if result != -1:
        print(f"Minimum number of steps to get exactly {d} liters: {result}")
    else:
        print(f"Not possible to measure exactly {d} liters using {m} and {n} liter jugs.")


#DEPTH FIRST SEARCH

def water_jug_dfs(capacity1, capacity2, target):
    visited = set()
    path = []

    def dfs(jug1, jug2):
        if (jug1, jug2) in visited:
            return False

        visited.add((jug1, jug2))
        path.append((jug1, jug2))

        if jug1 == target or jug2 == target:
            return True

        if dfs(jug1, capacity2):
            return True
        if dfs(capacity1, jug2):
            return True
        if dfs(0, jug2):
            return True
        if dfs(jug1, 0):
            return True
        if dfs(max(0, jug1 - (capacity2 - jug2)), min(capacity2, jug1 + jug2)):
            return True
        if dfs(min(capacity1, jug1 + jug2), max(0, jug2 - (capacity1 - jug1))):
            return True

        path.pop()
        return False

    dfs(0, 0)
    return path


capacity1 = 3
capacity2 = 5
target = 4

solution = water_jug_dfs(capacity1, capacity2, target)

if solution:
    print("Solution steps:")
    for step in solution:
        print(step)
else:
    print("No solution found.")
