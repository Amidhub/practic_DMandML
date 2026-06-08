import math
import random

# Проверка числа на простоту
def is_prime(n):
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# Поиск простого числа в диапазоне
def get_prime(min_val, max_val):
    while True:
        num = random.randint(min_val, max_val)
        if is_prime(num):
            return num

# Расширенный алгоритм Евклида
def egcd(a, b):
    if b == 0:
        return a, 1, 0
    g, x1, y1 = egcd(b, a % b)
    return g, y1, x1 - (a // b) * y1

# Обратный элемент по модулю
def mod_inv(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
        return None
    return x % m

# Генерация ключей RSA
def gen_keys():
    # берем простые числа поменьше
    p = get_prime(100, 300)
    q = get_prime(100, 300)
    while p == q:
        q = get_prime(100, 300)
    
    n = p * q
    phi = (p - 1) * (q - 1)
    
    e = 17
    while math.gcd(e, phi) != 1:
        e += 2
    
    d = mod_inv(e, phi)
    
    return (e, n), (d, n), p, q

# Шифрование строки
def encrypt(text, pub_key):
    e, n = pub_key
    res = []
    for ch in text:
        code = ord(ch)
        res.append(pow(code, e, n))
    return res

# Расшифрование
def decrypt(codes, priv_key):
    d, n = priv_key
    res = []
    for c in codes:
        res.append(chr(pow(c, d, n)))
    return ''.join(res)


# Основная часть
print("=" * 50)
print("RSA шифрование")
print("=" * 50)

# генерация ключей
pub, priv, p, q = gen_keys()
print(f"Простые числа: p = {p}, q = {q}")
print(f"Модуль n = {pub[1]}")
print(f"Открытая экспонента e = {pub[0]}")
print(f"Закрытая экспонента d = {priv[0]}")

# исходный текст
text = "Четные числа питательны, а нечетные просто вкусные"
print(f"\nИсходный текст: {text}")

# шифруем
enc = encrypt(text, pub)
print(f"\nЗашифрованный текст (первые 5 чисел): {enc[:5]}...")
print(f"Всего чисел: {len(enc)}")

# расшифровываем
dec = decrypt(enc, priv)
print(f"\nРасшифрованный текст: {dec}")

# проверка
if text == dec:
    print("\nУспешно! Расшифровка совпала с оригиналом.")
else:
    print("\nОшибка!")