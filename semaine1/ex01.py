import numpy as np
import time

print('')
print('Tupple:')
tupple = ('mbaron', 48)
print(tupple)

print('')
print('Dict:')
dict = {"login1": 48, "login2":50}
print(dict)

print('')
print('Age:')
age = dict["login1"];
print(age)

print('')
print('Dict after add mbaron:')
dict["mbaron"] = 48
print(dict)

print('')
print('List:')
A = [1, 3, 2, 7, 4, 10, 46]
print(A)

print('')
print('3 first elements:')
print(A[0:3])

print('')
print('B = 4 to 6 elements of A:')
B = A[3:6]
print(B)

print('')
print('C = A + B:')
C = A + B
print(C)

print('')
print('Add 5 to A:')
A.append(5)
print(A)

print('')
print('Add None to C:')
C.append(None)
print(C)

def concat(lst, n = 2):
    return n * lst

print('')
print('Function concat *n (n = 3):')
print(concat(B, 3));

print('')
print('Function concat *n (n default):')
print(concat(B));

print('')
print('Boucle while until None:')
i = 0
while(C[i] != None):
    print(C[i])
    i = i + 1

print('')
print('Nb odd numbers in A:')
nb = 0
for a in A:
    if a % 2 == 0:
        nb = nb + 1

print(nb)

print('')
print('C = only odd numbers of A:')
C = []
for a in A:
    if a % 2 == 0:
        C.append(a)

print(C)

print('')
print('create a & b (np.array):')
a = np.array(A);
b = np.array(B);
print(a)
print(b)

print('')
print('c = a + b :')
c = np.concatenate((a,b))
print(c)

print('')
print('divide by 3:')
c = c / 3
print(c)

print('')
print('sum:')
n = c.sum()
print(n)

print('')
print('create M & N:')
M = np.random.randint(10, size=(4,4))
N = np.random.randint(10, size=(4,4))
print(M)
print(N)

print('')
print('O = M + N:')
O = M + N
print(O)

print('')
print('P = M * N:')
start_time = time.clock()
P = np.matmul(M, N)
print(time.clock() - start_time, "seconds")
print(P)

print('')
print('Matrix None:')
Q = np.empty([4,4])
print(Q)
start_time = time.clock()
for i in range(4):
    for j in range(4):
        Q[i][j] = 0
        for k in range(4):
            Q[i][j] += M[i][k] * N[k][j]
            
print(time.clock() - start_time, "seconds")
print(Q)

def compare_matmul(s):
    print('')
    print('Big matrix ', s, ' :')
    M = np.random.randint(10, size=(s,s))
    N = np.random.randint(10, size=(s,s))
    start_time = time.clock()
    P = np.matmul(M, N)
    print("matmul", time.clock() - start_time, "seconds")
    Q = np.empty([s,s])
    start_time = time.clock()
    for i in range(s):
        for j in range(s):
            Q[i][j] = 0
            for k in range(s):
                Q[i][j] += M[i][k] * N[k][j]
            
    print(time.clock() - start_time, "seconds")
    
compare_matmul(10)
compare_matmul(100)
compare_matmul(200)
