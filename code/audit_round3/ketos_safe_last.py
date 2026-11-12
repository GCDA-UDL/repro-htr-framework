#!/usr/bin/env python
"""Audit round 3: same as code/scripts/ketos_safe.py (torch.load allowlist for the kraken VGSL config classes) plus
`save_last=True` on Lightning's ModelCheckpoint, so that a fixed-length run (`-q fixed -N <n>`) always keeps the
checkpoint of its final epoch as `last.ckpt`. Kraken 7.0.2 keeps only the 10 best checkpoints by validation metric
(kraken/ketos/recognition.py:195), which does not guarantee that the last epoch survives. Nothing else changes."""
import inspect
import sys

import torch
import kraken.configs.vgsl as kvgsl
import lightning.pytorch.callbacks as plc

safe = [obj for _, obj in inspect.getmembers(kvgsl, inspect.isclass)]
torch.serialization.add_safe_globals(safe)
print(f'[ketos_safe_last] allowlisted {len(safe)} classes from kraken.configs.vgsl; ModelCheckpoint(save_last=True)',
      file=sys.stderr)

_orig_init = plc.ModelCheckpoint.__init__


def _init(self, *args, **kwargs):
    kwargs.setdefault('save_last', True)
    _orig_init(self, *args, **kwargs)


plc.ModelCheckpoint.__init__ = _init

from kraken.ketos import cli  # noqa: E402

if __name__ == '__main__':
    sys.exit(cli())
