# Повороты
p0 = [1,2,3,4,5,6]
p1 = [2,3,4,5,6,1]
p2 = [3,4,5,6,1,2]
p3 = [4,5,6,1,2,3]
p4 = [5,6,1,2,3,4]
p5 = [6,1,2,3,4,5]

# Отражения
s1 = [1,6,5,4,3,2]
s2 = [2,1,6,5,4,3]
s3 = [3,2,1,6,5,4]
s4 = [4,3,2,1,6,5]
s5 = [5,4,3,2,1,6]
s6 = [6,5,4,3,2,1]

G = [p0,p1,p2,p3,p4,p5,s1,s2,s3,s4,s5,s6]
name = ["e","a","a2","a3","a4","a5","b1","b2","b3","b4","b5","b6"]

def mul(x,y):
    res = [0]*6
    for i in range(6):
        res[i] = x[y[i]-1]
    return res

def find(p):
    for i in range(12):
        if G[i] == p:
            return name[i]
    return "?"

print("   ", " ".join(name))
for i in range(12):
    row = []
    for j in range(12):
        row.append(find(mul(G[i], G[j])))
    print(name[i], " ".join(row))

print("\nПорядок группы: 12")

abel = True
for i in range(12):
    for j in range(12):
        if mul(G[i], G[j]) != mul(G[j], G[i]):
            abel = False
            break
    if not abel:
        break

if abel:
    print("Группа абелева")
else:
    print("Группа не абелева")
    