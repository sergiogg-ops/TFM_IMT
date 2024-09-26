'''
Obtains the mean of the first and second columns of a file.

Usage:
    python utils/get_means.py <file>
'''
from sys import argv

m1, m2 = [], []
with open(argv[1], 'r') as f:
    for line in f.readlines()[1:]:
        line = line.split()
        m1.append(float(line[0]))
        m2.append(float(line[1]))

print(sum(m1)/len(m1), sum(m2)/len(m2))
