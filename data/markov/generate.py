import sys
import numpy as np
import math
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-p", "--prob", default=0.5)
parser.add_argument("-l", "--len", default=100)
args = parser.parse_args()
config = vars(args)
# print(config)

p = float(config['prob'])
l = int(config['len'])

chars = [0, 1, 2, 3]
transitions = [[0.25, 0.25, 0, 0.5], [0.25, 0.25, 0.5, 0], [0.5, 0, 0.1, 0.4], [0, 0.5, 0.4, 0.1]]
def transition(c):
    return np.random.choice(chars, p=transitions[c])

f = open('markov_output.txt', 'w')

print(p, np.random.rand())
n = 1000
c = 0
for i in range(n):
    for j in range(l):
        c = transition(c)
        f.write(str(c))
    f.write('\n')