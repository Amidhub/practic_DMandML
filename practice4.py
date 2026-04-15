# Разложение рационального числа a/b в цепную дробь
def drob(a, b):
    res = []
    while b != 0:
        res.append(a // b)
        a, b = b, a % b
    return res

print(drob(297, 210))
print(drob(13, 8))
print(drob(1685546, 34789))

# Разложение квадратичной иррациональности в периодическую цепную дробь
def irr(x):
    n = round(x * x)
    a0 = int(x)
    m = 0
    d = 1
    a = a0
    res = [a0]
    while a != 2 * a0:
        m = d * a - m
        d = (n - m * m) // d
        a = int((a0 + m) // d)
        res.append(a)
    return res

import math
print(irr(math.sqrt(2)))
print(irr(math.sqrt(23)))
print(irr(math.sqrt(1287)))