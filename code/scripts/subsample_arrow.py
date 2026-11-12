#!/usr/bin/env python
import sys, random
import pyarrow.ipc as ipc

src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

# First pass: count rows and record schema
with open(src, 'rb') as f:
    reader = ipc.open_file(f)
    schema = reader.schema
    nb = reader.num_record_batches
    batch_sizes = [reader.get_batch(i).num_rows for i in range(nb)]

total = sum(batch_sizes)
if n > total:
    print(f'ERROR: demanades {n} pero el fitxer en te {total}')
    sys.exit(1)

random.seed(seed)
chosen = set(random.sample(range(total), n))

# Second pass: stream batches, keep only chosen rows, write incrementally
written = 0
with open(src, 'rb') as f, open(dst, 'wb') as fout:
    reader = ipc.open_file(f)
    writer = ipc.new_file(fout, schema)
    offset = 0
    for i in range(nb):
        batch = reader.get_batch(i)
        local = [r - offset for r in range(offset, offset + batch.num_rows) if r in chosen]
        if local:
            sub = batch.take(local)
            writer.write_batch(sub)
            written += len(local)
        offset += batch.num_rows
    writer.close()

print(f'{src}: {total} -> {dst}: {written} rows (seed {seed})')
