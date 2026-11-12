#!/usr/bin/env python
import sys
import inspect
import torch
import kraken.configs.vgsl as kvgsl

# Allowlist all classes defined in kraken's VGSL config module.
# Justification: checkpoints are locally generated and trusted; PyTorch 2.6+
# defaults torch.load(weights_only=True), which rejects these dataclasses.
safe = [obj for _, obj in inspect.getmembers(kvgsl, inspect.isclass)]
torch.serialization.add_safe_globals(safe)
print(f'[ketos_safe] allowlisted {len(safe)} classes from kraken.configs.vgsl',
      file=sys.stderr)

from kraken.ketos import cli

if __name__ == '__main__':
    sys.exit(cli())
