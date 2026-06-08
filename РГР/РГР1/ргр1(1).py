def print_matrix(name, matrix):
    print(f"\n{name}:")
    for row in matrix:
        print(row)

def disjunction(A, B):
    n, m = len(A), len(A[0])
    C = [[0]*m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            C[i][j] = 1 if (A[i][j] or B[i][j]) else 0
    return C

def transpose(A):
    n, m = len(A), len(A[0])
    return [[A[i][j] for i in range(n)] for j in range(m)]

def invert(A):
    n, m = len(A), len(A[0])
    return [[1 - A[i][j] for j in range(m)] for i in range(n)]

def subtraction(A, B):
    n, m = len(A), len(A[0])
    C = [[0]*m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            C[i][j] = 1 if (A[i][j] and (1 - B[i][j])) else 0
    return C

def multiplication(A, B):
    n, m = len(A), len(B[0])
    k = len(A[0])
    C = [[0]*m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for t in range(k):
                if A[i][t] and B[t][j]:
                    C[i][j] = 1
                    break
    return C

R1 = [[0,1,1,0,0,0,0,0],[0,0,1,1,0,0,0,0]]
R2 = [[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0]]

print("R1:", R1)
print("R2:", R2)

print("\n1. Дизъюнкция:")
print_matrix("R1 ∨ R2", disjunction(R1, R2))

print("\n2. Транспонирование:")
print_matrix("R1^T", transpose(R1))

print("\n3. Инвертирование:")
print_matrix("¬R1", invert(R1))

print("\n4. Вычитание:")
print_matrix("R1 - R2", subtraction(R1, R2))

print("\n5. Умножение:")
print_matrix("R1 × R1^T", multiplication(R1, transpose(R1)))