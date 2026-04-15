# # import numpy as np
# # def boolean_matrix_multiplication(A, B):
# #     # Применяем логическое умножение строк и столбцов матриц с помощью операций AND и OR
# #     return np.logical_and.reduce(A[:, :, np.newaxis] | B[np.newaxis, :, :], axis=1)

# # # Пример использования
# # A = np.array([[0, 0], [0, 1]])
# # B = np.array([[1, 0], [1, 0]])
# # C = boolean_matrix_multiplication(A, B)
# # print(C)


# from scipy.sparse import csr_array
# from scipy.sparse.csgraph import floyd_warshall
# import networkx as nx
# import matplotlib.pyplot as plt
# import numpy as np
# # Создание графа
# G = nx.Graph() #не ориентированный
# G=nx.DiGraph(directed=True) # ориентированный

# # Добавление вершин, т.е. задание множества A для прямого произведения A*A
# G.add_nodes_from([1, 2, 3, 4, 5])
# # Добавление рёбер, то есть задание бинарного отношения, т.е. подмножества A*A
# # A=[(1, 3), (2, 3), (2, 1),(5,4),(1,5),(2,3)]
# A=[(1,2), (1,3), (2,3)]

# G.add_edges_from(A)

# # Визуализация графа
# # nx.draw(G, with_labels=True, node_color='lightblue')
# # plt.show()

# #Нахождение матрицы смежности B по бинарному отношению
# n=3#задание числа вершин графа
# B = np.zeros((n, n))

# for t in A:
#     B[t[0]-1][t[1]-1]=1
# #матрица смежности графа по заданному бинарному отношению
# print(B)

# #Проверка на рефлексивность
# def refl(B):
#     if all([B[i][i]==1 for i in range(len(B))]):
#         return 'Yes'
#     return 'No'
# print(f'Рефлексивность: {refl(B)}')

# #Проверка на антирефлексивность
# def antirefl(B):
#     if all([B[i][i]==0 for i in range(len(B))]):
#         return 'Yes'
#     return 'No'

# print(f'Антирефлексивность: {antirefl(B)}')


# #Проверка на Симметричность
# def simetr(B):
#     if all([B[i][j]==B[j][i] for i in range(len(B)) for j in range(len(B))]):
#         return 'Yes'
#     return 'No'
# print(f'Симметричность: {simetr(B)}')


# #Проверка на Антисимметричность
# def antisimetr(B):
#     if all([(B[i][j]==1 and B[j][i]==0 or B[i][j]==0 and B[j][i]==1 or B[i][j]==0 and B[j][i]==0) for i in range(len(B)) for j in range(len(B)) if i!=j] ):
#         return 'Yes'
#     return 'No'
# print(f'Антисимметричность: {antisimetr(B)}')

# #Проверка на Транзитивность
# def transitivnosti(B):
#     n = len(B)
#     for i in range(n):
#         for j in range(n):
#             for k in range(n):
#                 if B[i][j] == 1 and B[j][k] == 1 and B[i][k] == 0:
#                     return 'No'
#     return 'Yes'
# print(f'Транзитивность: {transitivnosti(B)}')

# #Проверка на линейность
# def lineinosti(B):
#     n = len(B)
#     for i in range(n):
#         for j in range(n):
#             if i != j and B[i][j] == 0 and B[j][i] == 0:
#                 return 'No'
#     return 'Yes'
# print(f'Линейность: {lineinosti(B)}')


# graph = B
# graph = csr_array(graph)
# print(graph)


# dist_matrix, predecessors = floyd_warshall(csgraph=graph, directed=True, return_predecessors=True)
# print(dist_matrix) # Матрица расстояний N x N между узлами графа. dist_matrix[i,j] задает кратчайшее расстояние от точки i до точки j на графе
# print(predecessors)


# По заданным бинарным отношениям строятся графы объединения, пересечения и композиции бинарных отношений


'''
from scipy.sparse import csr_array
from scipy.sparse.csgraph import floyd_warshall
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def bul_umn(M1, M2):# булево умножение двух матриц M1 и M2
    A = np.array(M1)
    B = np.array(M2)
    C = (A @ B > 0).astype(int)
    return C

def bul_slozh(M1, M2):
    A = np.array(M1)
    B = np.array(M2)
    C = (A + B > 0).astype(int)
    return C


k=[1, 2, 3]

r1=[(1,1),(2,2), (3,3), (1,2), (2,3)]
r2=[(3,2),(1,3),(2,1), (2,2)]


n = 3
BB = np.zeros((n, n))
CC = np.zeros((n, n))

for t in r1:
    BB[t[0] - 1][t[1] - 1] = 1

for t in r2:
    CC[t[0] - 1][t[1] - 1] = 1

print("Матрица смежности графа G1", BB)
print("Матрица смежности графа G2", CC)
N3 = bul_umn(BB, CC)
print("Матрица смежности графа композиции G1 и G2", N3)
N4 = bul_slozh(BB, CC)

print("Матрица смежности графа объединения G1 и G2", N4)
N5 = BB*CC
print("Матрица смежности графа пересечения G1 и G2", N5)


G1 = nx.DiGraph(np.matrix(BB))
G2 = nx.DiGraph(np.matrix(CC))
G3 = nx.DiGraph(np.matrix(N3))
G4 = nx.DiGraph(np.matrix(N4))
G5 = nx.DiGraph(np.matrix(N5))

pos1 = nx.spring_layout(G1, scale=100000,center=[0, 0])
pos2 = nx.circular_layout(G2, scale=80000,center=[30, 40])
pos3 = nx.spectral_layout(G3, scale=50000,center=[300, 400])
pos4 = nx.spiral_layout(G4, scale=40000,center=[35, 4])
pos5 = nx.spring_layout(G5, scale=60000,center=[3000, 477])


nx.draw(G1, pos=pos1, with_labels=True, node_size=200, arrows=True, node_color="blue",font_size=10,font_weight="bold")
nx.draw(G2, pos=pos2,with_labels=True, node_size=200, arrows=True, node_color="lightblue",font_size=10,font_weight="bold")
#nx.draw(G3, with_labels=True, pos=pos3, node_size=200, arrows=True, node_color="red",font_size=10,font_weight="bold")
nx.draw(G4, with_labels=True, node_size=200,pos=pos4, arrows=True, node_color="orange",font_size=10,font_weight="bold")
#nx.draw(G5, with_labels=True, node_size=200,pos=pos5, arrows=True,node_color="green",font_size=10,font_weight="bold")
plt.show()
'''




# import numpy as np
# from hassediagram import plot_hasse

# data = (np.array([
#     [0, 1, 1, 1, 0,0,0,0],
#     [0, 0, 0, 0, 1,1,0,0],
#     [0, 0, 0, 0, 1,0,1,0],
#     [0, 0, 0, 0, 0,1,1,0],
#     [0, 0, 0, 0, 0,0,0,1],
#     [0, 0, 0, 0, 0,0,0,1],
#     [0, 0, 0, 0, 0,0,0,1],
#     [0, 0, 0, 0, 0,0,0,0],

# ]))
# # labels = ["node a", "node b", "node c", "node d", "node e"]
# plot_hasse(data)


#Хасса для делимости 
import numpy as np
from hassediagram import plot_hasse

data = (np.array([
    [0,1,1,0,0,0,0,0],
    [0,0,0,1,1,0,0,0], 
    [0,0,0,0,1,0,0,0], 
    [0,0,0,0,0,1,1,0],
    [0,0,0,0,0,0,1,1],
    [0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0],
]))

data_reversed = data.T
labels = ["1", "2", "3", "4", "6", "8", "12", "24"]
plot_hasse(data, labels)

