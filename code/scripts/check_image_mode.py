#!/usr/bin/env python
import sys, io, collections
import numpy as np
import pyarrow.ipc as ipc
from PIL import Image

path = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 200
modes = collections.Counter()
n = 0
with open(path, 'rb') as f:
    reader = ipc.open_file(f)
    for i in range(reader.num_record_batches):
        batch = reader.get_batch(i)
        for j in range(batch.num_rows):
            row = batch.column('lines')[j].as_py()
            raw = row['im']
            if isinstance(raw, bytes):
                im = Image.open(io.BytesIO(raw))
            else:
                im = Image.fromarray(np.array(raw, dtype=np.uint8))
            modes[im.mode] += 1
            n += 1
            if n >= limit:
                break
        if n >= limit:
            break
print(f'{path}: {n} images sampled')
for m, c in modes.most_common():
    print(f'  mode {m}: {c}')
