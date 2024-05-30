import sys, math
import numpy as np

pattern1 = '23'
markov = {}
size = 2
rng = np.random.default_rng(42)
for s in range(1, size+1):
    # print(s)
    for i in range(2**s):
        curr_seed = format(i, f'0{s}b')
        # print(i, curr_seed)
        bias = rng.choice([0.25, 0.75])
        markov[curr_seed] = bias
    # print()
print(markov)

p = float(sys.argv[2])
num_samples = int(sys.argv[1])
length = 100
f = open('data.txt', 'w')
f.write('\n')
for i in range(num_samples):
    curr = []
    if rng.random() < p:
        curr = ['0' if rng.random() < 0.5 else '1']
        while len(curr) < 100:
            bias = markov[''.join(curr[-size:])]
            curr.append('0' if rng.random() < bias else '1')
    else:
        for i in range(100//len(pattern1)):
            curr.append(pattern1)
    f.write(''.join(curr))
    f.write('\n')
f.close()