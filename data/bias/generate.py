import numpy as np
import argparse

biases = [0.25, 0.5, 0.75]
parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-n", "--num_lines", default=100)
parser.add_argument("-l", "--line_len", default=100)
args = parser.parse_args()
config = vars(args)
# print(config)

num_lines = int(config['num_lines'])
line_len = int(config['line_len'])

bias_enc = {0.25:'a', 0.5:'b', 0.75:'c'}
f = open('bias_output.txt', 'w')
f.write("\n")
for _ in range(num_lines):
    bias_i = np.random.choice([i for i in range(len(biases))])
    bias = biases[bias_i]
    enc = bias_enc[bias]
    l = []
    for _ in range(line_len):
        if np.random.rand() < bias:
            l.append('0')
        else:
            l.append('1')
    s = f'{enc}:' + ''.join(l) + '\n'
    # s = ''.join(l) + '.\n'
    f.write(s)
f.close()

