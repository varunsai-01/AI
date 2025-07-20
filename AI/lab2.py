# best first search

def best_fs(start, goal):
    open_set = {start}
    parents = {start: None}

    while open_set:
        current = min(open_set, key=heuristic)
        
        if current == goal:
            return build_path(parents, goal)
        
        open_set.remove(current)

        for neighbor, _ in Graph_nodes.get(current, []):
            if neighbor not in parents:
                open_set.add(neighbor)
                parents[neighbor] = current

    print("Path does not exist!")
    return None


def build_path(parents, node):
    path = []
    while node is not None:
        path.append(node)
        node = parents[node]
    path.reverse()
    print(f"Path found: {path}")
    return path


def heuristic(node):
    H_dist = {'A':10, 'B':8, 'C':5, 'D':7, 'E':3, 'F':6, 
              'G':4, 'H':2, 'I':1, 'J':0}
    return H_dist.get(node, float('inf'))


Graph_nodes = {
    'A': [('B', 4), ('C', 3)],
    'B': [('D', 5), ('E', 6)],
    'C': [('F', 2)],
    'D': [('G', 2)],
    'E': [('H', 3)],
    'F': [('I', 4)],
    'G': [('J', 6)],
    'H': [('J', 2)],
    'I': [('J', 1)],
    'J': []
}

best_fs('A', 'J')



# A* search

def aStarAlgo(start_node, stop_node):
    open_set = set([start_node])
    closed_set = set()
    g = {start_node: 0}  # Distance from start node
    parents = {start_node: start_node}  # Parent map

    while open_set:
        n = min(open_set, key=lambda v: g[v] + heuristic(v))

        if n == stop_node:
            path = []
            while parents[n] != n:
                path.append(n)
                n = parents[n]
            path.append(start_node)
            path.reverse()
            print('Path found: {}'.format(path))
            return path

        open_set.remove(n)
        closed_set.add(n)

        for (m, weight) in get_neighbors(n):
            if m in closed_set:
                continue

            tentative_g = g[n] + weight
            if m not in open_set or tentative_g < g.get(m, float('inf')):
                g[m] = tentative_g
                parents[m] = n
                open_set.add(m)

    print('Path does not exist!')
    return None

def get_neighbors(v):
    if v in Graph_nodes:
        return Graph_nodes[v]
    
    else:
        return None
    

def heuristic(n):
    H_dist = {'A':10, 'B':8, 'C':5, 'D':7, 'E':3, 'F':6, 
              'G':4, 'H':2, 'I':1, 'J':0}
    return H_dist[n]


Graph_nodes = {
    'A': [('B', 4), ('C', 3)],
    'B': [('D', 5), ('E', 6)],
    'C': [('F', 2)],
    'D': [('G', 2)],
    'E': [('H', 3)],
    'F': [('I', 4)],
    'G': [('J', 6)],
    'H': [('J', 2)],
    'I': [('J', 1)],
    'J': []
}

aStarAlgo('A', 'J')