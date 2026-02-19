# import numpy as np
# def boolean_matrix_multiplication(A, B):
#     # Применяем логическое умножение строк и столбцов матриц с помощью операций AND и OR
#     return np.logical_and.reduce(A[:, :, np.newaxis] | B[np.newaxis, :, :], axis=1)

# # Пример использования
# A = np.array([[0, 0], [0, 1]])
# B = np.array([[1, 0], [1, 0]])
# C = boolean_matrix_multiplication(A, B)
# print(C)


from scipy.sparse import csr_array
from scipy.sparse.csgraph import floyd_warshall
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
# Создание графа
G = nx.Graph() #не ориентированный
G=nx.DiGraph(directed=True) # ориентированный

# Добавление вершин, т.е. задание множества A для прямого произведения A*A
G.add_nodes_from([1, 2, 3, 4, 5])
# Добавление рёбер, то есть задание бинарного отношения, т.е. подмножества A*A
# A=[(1, 3), (2, 3), (2, 1),(5,4),(1,5),(2,3)]
A=[(1,2), (1,3), (2,3)]

G.add_edges_from(A)

# Визуализация графа
# nx.draw(G, with_labels=True, node_color='lightblue')
# plt.show()

#Нахождение матрицы смежности B по бинарному отношению
n=3#задание числа вершин графа
B = np.zeros((n, n))

for t in A:
    B[t[0]-1][t[1]-1]=1
#матрица смежности графа по заданному бинарному отношению
print(B)

#Проверка на рефлексивность
def refl(B):
    if all([B[i][i]==1 for i in range(len(B))]):
        return 'Yes'
    return 'No'
print(f'Рефлексивность: {refl(B)}')

#Проверка на антирефлексивность
def antirefl(B):
    if all([B[i][i]==0 for i in range(len(B))]):
        return 'Yes'
    return 'No'

print(f'Антирефлексивность: {antirefl(B)}')


#Проверка на Симметричность
def simetr(B):
    if all([B[i][j]==B[j][i] for i in range(len(B)) for j in range(len(B))]):
        return 'Yes'
    return 'No'
print(f'Симметричность: {simetr(B)}')


#Проверка на Антисимметричность
def antisimetr(B):
    if all([(B[i][j]==1 and B[j][i]==0 or B[i][j]==0 and B[j][i]==1 or B[i][j]==0 and B[j][i]==0) for i in range(len(B)) for j in range(len(B)) if i!=j] ):
        return 'Yes'
    return 'No'
print(f'Антисимметричность: {antisimetr(B)}')

#Проверка на Транзитивность
def transitivnosti(B):
    n = len(B)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if B[i][j] == 1 and B[j][k] == 1 and B[i][k] == 0:
                    return 'No'
    return 'Yes'
print(f'Транзитивность: {transitivnosti(B)}')

#Проверка на линейность
def lineinosti(B):
    n = len(B)
    for i in range(n):
        for j in range(n):
            if i != j and B[i][j] == 0 and B[j][i] == 0:
                return 'No'
    return 'Yes'
print(f'Линейность: {lineinosti(B)}')


graph = B
graph = csr_array(graph)
print(graph)


dist_matrix, predecessors = floyd_warshall(csgraph=graph, directed=True, return_predecessors=True)
print(dist_matrix) # Матрица расстояний N x N между узлами графа. dist_matrix[i,j] задает кратчайшее расстояние от точки i до точки j на графе
print(predecessors)


