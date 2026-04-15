def topological_sort(vertices, edges):
    graph = {}
    in_degree = {}
    
    for v in vertices:
        graph[v] = []
        in_degree[v] = 0
    
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1
    
    queue = [v for v in vertices if in_degree[v] == 0]
    linear_order = []
    
    while queue:
        node = queue.pop(0)
        linear_order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    if len(linear_order) != len(vertices):
        return None
    
    return linear_order

vertices = [1, 2, 3, 4, 5, 6]
edges = [(1,2), (2,3), (3,4), (3,5), (5,6)]

result = topological_sort(vertices, edges)
print("Вершины:", vertices)
print("Отношения порядка:", edges)
print("Линейный порядок:", result)