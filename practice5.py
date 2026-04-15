# Линейное диофантово уравнение ax + by = c
# c должно быть равно НОД(a,b)

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def chain(a, b):
    res = []
    while b:
        res.append(a // b)
        a, b = b, a % b
    return res

def solve(a, b, c):
    if c != gcd(a, b):
        return "Нет решений"
    
    A, B = a, b
    frac = chain(a, b)
    
    num = 1
    den = frac[-2]
    
    for i in range(len(frac) - 3, -1, -1):
        num = frac[i] * den + num
        num, den = den, num
    
    x0 = num
    y0 = -den
    
    return f"x = {x0} + {B}k, y = {y0} - {A}k"

# Примеры
print(solve(99, 70, 1))
print(solve(6, 8, 2))
print(solve(17, 5, 1))
