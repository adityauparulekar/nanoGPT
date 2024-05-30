check_string = '1011011000101101100010110110001011011000101101100010110110001011011000101101100010110110001011011000'
f = open('rd_samples.txt')
l = list(map(lambda x: x.strip(), f.read().split("---------------")[:-1]))
# print(l)

def rot_s(s, i):
    i %= len(s)
    return s[i:] + s[:i]
def check_rot(a, b):
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if rot_s(a, i) == b:
            return True
    return False
count = 0
count_rot = 0
count_nl = 0
count_else = 0
count_len = 0
for line in l:
    if line == check_string:
        count += 1
    elif check_rot(line, check_string):
        count_rot+=1
    else:
        if '\n' in line:
            count_nl += 1
        elif len(line) != len(check_string):
            count_len += 1
        else:
            count_else += 1
            print(line)
print('correct \t', count, '/', len(l))
print('rotations\t', count_rot, '/', len(l))
print('newlines\t', count_nl+count_len, '/', len(l))
print('else\t\t', count_else, '/', len(l))