import numpy as np
import sys, re, math, os

original_markov = {}
size = 5
rng = np.random.default_rng(42)
for s in range(1, size+1):
    # print(s)
    for i in range(2**s):
        curr_seed = format(i, f'0{s}b')
        # print(i, curr_seed)
        bias = rng.random()
        original_markov[curr_seed] = bias
    # print()
# print(markov)

n = int(sys.argv[1])
p = float(sys.argv[2])
iter = int(sys.argv[3])

def calculate01(l, m, history_size=5):
    for i in range(len(l)-1):
        pref = l[max(0, i-history_size+1):i+1]
        if pref not in m:
            m[pref] = [0, 0]
        if l[i+1] == '0':
            m[pref][0]+=1
        else:
            m[pref][1]+=1
def calculate23(l, m, history_size=5):
    for i in range(len(l)-1):
        pref = l[max(0, i-history_size+1):i+1]
        if pref not in m:
            m[pref] = [0, 0]
        if l[i+1] == '2':
            m[pref][0]+=1
        else:
            m[pref][1]+=1
f = open('rd_samples.txt')

lines = '\n'.join(f.readlines())

lines = list(map(lambda x:x.strip(), lines.split('---------------')))
distances01 = []
distances23 = []
num_len = 0
num_01s = 0
num_23s = 0
invalid = 0

markov01 = {}
markov23 = {}
for l in lines:
    first = '0' in l or '1' in l
    second = '2' in l or '3' in l

    if (first and second) or (not first and not second) or ('\n' in l):
        invalid+=1
    elif first and not second:
        num_01s += 1
        calculate01(l, markov01)
    elif second and not first:
        num_23s += 1
        calculate23(l, markov23)
    else:
        print("ERROR", l)
f.close()

avg_diff01 = -1
if len(markov01) > 0:
    avg_diff01 = 0
    count = 0
    for pref in markov01:
        if markov01[pref][0] + markov01[pref][1] > 50:
            bias = markov01[pref][0]/(markov01[pref][0] + markov01[pref][1])
            diff = original_markov[pref] - bias
            avg_diff01 += diff**2
            count+=1
    avg_diff01 /= count
    avg_diff01 = np.sqrt(avg_diff01)

avg_diff23 = -1
if len(markov23) > 0:
    avg_diff23 = 0
    for pref in markov23:
        bias = markov23[pref][0]/(markov23[pref][0] + markov23[pref][1])
        diff = (1 if pref[-1] == '3' else 0) - bias
        avg_diff23 += diff**2
    avg_diff23 /= len(markov23)
    avg_diff23 = np.sqrt(avg_diff23)

f = open('history.txt', 'a')
f.write(f'{iter}, {n}, {p}: num invalid: {invalid}, \n\t percent of 01-samples: {num_01s/(num_01s+num_23s+1)}, \n\t markov diff 01: {avg_diff01}, \n\t markov diff 23: {avg_diff23}\n')
f.write(str(markov01))
f.write('\n')
f.write(str(markov23))
f.write('\n')
f.close()
