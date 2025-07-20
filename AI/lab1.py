#breath first search

graph = {
    'A':['B','C'],
    'B':['D','E'],
    'C':['F'],
    'D':[],
    'E':['F'],  
    'F':[]
}    
visited = []
queue = []

def bfs(visited, graph, node):
    visited.append(node)
    queue.append(node)

    while queue:
        s = queue.pop(0)
        print(s, end=" ")

        for neighbour in graph[s]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)

bfs(visited, graph, 'A')                


#deapth first search

graph = {
    'A':['B','C'],
    'B':['D','E'],
    'C':['F'],
    'D':[],
    'E':['F'],  
    'F':[]
} 

visited = []

def dfs(visited,graph,node):
    visited.append(node)
    queue.append(node)
    print(node,end="-")

    for neighbour in graph[node]:
        if neighbour is not visited:
            dfs(visited,graph,neighbour)

dfs(visited,graph,'A')

#uniform cost search

import heapq

graph = {
    'A':[("B",1),('C',1)],
    'B':[("D",1),('E',2)],
    'C':[("F",5)],
    'D':[],
    'E':[('F',1)],
    'F':[]
} 

def ucs(graph, start, goal):
    queue = [(0, start)]
    visited = set()
    cost = {start: 0}
    parent = {start: None}

    while queue:
        current_cost, node = heapq.heappop(queue)

        if node in visited:
            continue

        visited.add(node)
        print(f"Visited node {node}, current cost {current_cost}")

        if node == goal:
            break

        for neighbour, neighbour_cost in graph.get(node, []):
            if neighbour not in visited:
                new_cost = current_cost + neighbour_cost

                if neighbour not in cost or new_cost < cost[neighbour]:
                    cost[neighbour] = new_cost
                    heapq.heappush(queue, (new_cost, neighbour))
                    parent[neighbour] = node

    # Reconstruct the path
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()

    return path, cost.get(goal, float('inf'))

path, total_cost = ucs(graph, "A", "F")
print(f"Path to F: {'->'.join(path)}")
print(f"Total cost: {total_cost}")

      
      
# depth limited search

graph = {
    'A':['B','C'],
    'B':['D','E'],
    'C':['F'],
    'D':[],
    'E':['F'],  
    'F':[]
} 

def depth_limited_search(graph, node, limit, depth=0):
    if depth > limit:
        return
    
    print(f"Visiting node:{node}, Depth:{depth}")

    for neighbour in graph[node]:
        depth_limited_search(graph, neighbour, limit, depth+1)

depth_limit = 2
depth_limited_search(graph,'A', depth_limit)
