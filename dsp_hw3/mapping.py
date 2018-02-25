import sys
import collections
mapping = collections.defaultdict(list)

for l in open(sys.argv[1], 'r', encoding='big5hkscs'):

    big5 = l[0]
    zhuyin = l[2:].split('/')

    mapping[big5].append(big5)

    for z in zhuyin:
        mapping[ z[0] ].append(big5)


with open(sys.argv[2], 'w', encoding='big5hkscs') as f:
    for b, z in mapping.items():
        f.write(b + ' ' + ' '.join(z) + '\n')

