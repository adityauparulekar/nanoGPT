import numpy as np
import sys, re, math, os

n = int(sys.argv[1])
p = float(sys.argv[2])
iter = int(sys.argv[3])
def levenshteinDistance(A, B):
    N, M = len(A), len(B)
    # Create an array of size NxM
    dp = [[0 for i in range(M + 1)] for j in range(N + 1)]

    # Base Case: When N = 0
    for j in range(M + 1):
        dp[0][j] = j
    # Base Case: When M = 0
    for i in range(N + 1):
        dp[i][0] = i
    # Transitions
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            if A[i - 1] == B[j - 1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j], # Insertion
                    dp[i][j-1], # Deletion
                    dp[i-1][j-1] # Replacement
                )

    return dp[N][M]

s01 = "0011000101110100000101011100000110111011000000010100110001011101000001010111000001101110110000000101"
s23 = "2323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323"
f = open('rd_samples.txt')

lines = '\n'.join(f.readlines())

lines = list(map(lambda x:x.strip(), lines.split('---------------')))
distances01 = []
distances23 = []
num_len = 0
num_01s = 0
num_23s = 0
invalid = 0
for l in lines:
    first = '0' in l or '1' in l
    second = '2' in l or '3' in l

    if (first and second) or (not first and not second):
        invalid+=1
    if first and not second:
        num_01s += 1
        d = levenshteinDistance(s01, l)
        distances01.append(d)
    if second and not first:
        num_23s += 1
        d = levenshteinDistance(s23, l)
        distances23.append(d)
f.close()
a01 = np.array(distances01)
if len(a01) > 0:
    average_d01 = np.mean(a01)
else:
    average_d01 = -1
a23 = np.array(distances23)
if len(a23) > 0:
    average_d23 = np.mean(a23)
else:
    average_d23 = -1

f = open('non_newline.txt', 'a')
f.write(f'{iter}, {n}, {p}: num invalid: {invalid}, \n\t percent of 01-samples: {num_01s/(num_01s+num_23s)}, \n\t average distance 01: {average_d01}, \n\t average distance 23: {average_d23}\n')
f.close()
