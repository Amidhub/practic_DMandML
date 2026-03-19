def cepnay_drob(a, b):
    lst = []
    while b != 0:
        lst.append(a // b)
        a, b = b, a % b
    return lst

# Примеры
print(cepnay_drob(75, 100))    # [0, 1, 3]
print(cepnay_drob(355, 113))   # [3, 7, 16]
print(cepnay_drob(22, 7))       # [3, 7]
print(cepnay_drob(13, 5))       # [2, 1, 1, 2]