biases = [0.25, 0.5, 0.75]
for bias in biases:
    f = open(f'{bias}_emp.txt', 'r')
    l = list(map(float, f.readline()[:-1].split(',')))
    print("TRUE: ", bias, "EMP: ", sum(l)/len(l))