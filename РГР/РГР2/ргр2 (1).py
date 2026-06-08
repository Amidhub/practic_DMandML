# Решение систем линейных сравнений

import math

# обратный элемент через расширенный алгоритм Евклида
def inv_mod(a, m):
    m0 = m
    y = 0
    x = 1
    if m == 1:
        return 0
    while a > 1:
        q = a // m
        t = m
        m = a % m
        a = t
        t = y
        y = x - q * y
        x = t
    if x < 0:
        x += m0
    return x

# решение системы через КТО (модули должны быть взаимно простыми)
def crt(rem, mod):
    if len(rem) != len(mod):
        print("Ошибка: разное количество")
        return None
    
    # проверка на взаимную простоту
    for i in range(len(mod)):
        for j in range(i + 1, len(mod)):
            if math.gcd(mod[i], mod[j]) != 1:
                print(f"Модули {mod[i]} и {mod[j]} не взаимно просты")
                return None
    
    N = 1
    for m in mod:
        N *= m
    
    res = 0
    for i in range(len(mod)):
        Ni = N // mod[i]
        inv = inv_mod(Ni, mod[i])
        res += rem[i] * Ni * inv
    
    return res % N, N

# универсальное решение (если модули не взаимно простые)
def crt_general(rem, mod):
    if len(rem) != len(mod):
        print("Ошибка: разное количество")
        return None
    
    x = rem[0]
    m = mod[0]
    
    for i in range(1, len(mod)):
        g = math.gcd(m, mod[i])
        
        if (rem[i] - x) % g != 0:
            print("Система не имеет решений")
            return None
        
        m1 = m // g
        m2 = mod[i] // g
        diff = (rem[i] - x) // g
        
        inv = inv_mod(m1, m2)
        t = (diff * inv) % m2
        m = m * m2
        x = (x + m // m2 * t) % m
    
    return x, m


# Примеры
print("Пример 1 (КТО):")
print("x ≡ 4 (mod 5)")
print("x ≡ 1 (mod 9)")
print("x ≡ 3 (mod 11)")
res = crt([4,1,3], [5,9,11])
if res:
    x, N = res
    print(f"Ответ: x = {x} + {N}k\n")

print("Пример 2 (общий случай):")
print("x ≡ 5 (mod 8)")
print("x ≡ 3 (mod 12)")
res = crt_general([5,3], [8,12])
if res:
    x, m = res
    print(f"Ответ: x ≡ {x} (mod {m})\n")
else:
    print()

print("Пример 3 (противоречие):")
print("x ≡ 2 (mod 6)")
print("x ≡ 5 (mod 9)")
res = crt_general([2,5], [6,9])
if res:
    x, m = res
    print(f"Ответ: x ≡ {x} (mod {m})\n")