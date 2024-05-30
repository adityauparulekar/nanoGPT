import sys, math
import numpy as np

pattern1 = '23'
pattern2 = '0011000101110100000101011100000110111011000000010100110001011101000001010111000001101110110000000101'
p = float(sys.argv[2])
num_samples = int(sys.argv[1])
length = 100
f = open('data.txt', 'w')
f.write('\n')
for i in range(num_samples):
    curr = []
    if np.random.rand() < p:
        for i in range(length//len(pattern2)):
            curr.append(pattern2)
    else:
        for i in range(length//len(pattern1)):
            curr.append(pattern1)
    f.write(''.join(curr))
    f.write('\n')
f.close()
