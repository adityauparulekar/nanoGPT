import sys

f = open('markov.txt', 'r')

lines = f.readlines()

a = 0
b = 0
c = 0
d = 0
for l in lines:
    if len(l) == 4 and l[0] in '01' and l[1] in '12':
        if l[0] == '0':
            if l[1] == '1':
                a += 1
            else:
                b += 1
        if l[0] == '1':
            if l[1] == '1':
                c += 1
            else:
                d += 1
print(a, b, c, d)
print(a/(a+b), b/(a+b), a+b, c/(c+d), d/(c+d), c+d)