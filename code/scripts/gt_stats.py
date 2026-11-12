#!/usr/bin/env python
import sys, collections
import pyarrow.ipc as ipc

path = sys.argv[1]
texts = []
with open(path, 'rb') as f:
    reader = ipc.open_file(f)
    for i in range(reader.num_record_batches):
        batch = reader.get_batch(i)
        for j in range(batch.num_rows):
            texts.append(batch.column('lines')[j].as_py()['text'])

total = len(texts)
uniq = len(set(texts))
print(f'{path}')
print(f'  total lines : {total}')
print(f'  unique texts: {uniq}  ({100*uniq/total:.1f}%)')
print(f'  most common:')
for t, c in collections.Counter(texts).most_common(5):
    print(f'    {c:5d}x  {t[:70]!r}')
