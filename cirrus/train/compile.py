"""Regional torch.compile for training.

Compiling the whole model trips an Inductor fusion bug in the backward
(torch 2.14 / triton 3.8: ``AssertionError: 4094 v.s. 4095`` in
``get_numel_rnumel``). The shared final norm and tied embedding receive
gradient from the main loss (T-1 positions) and from the MTP passes (T-2,
T-3), and Inductor tries to fuse those reductions of different lengths into
one kernel. Compiling each layer's pieces separately keeps the loss and MTP
head eager (they are mostly large matmuls already) and still fuses the many
small elementwise ops (fp32 norm casts, rotary, gates, SwiGLU, GDN gating)
where eager mode loses its time. The math is unchanged.
"""

from __future__ import annotations

import torch.nn as nn


def compile_layers(model: nn.Module, **kwargs) -> int:
    """Compile, in place, the norms, mixer and FFN of every self- and cross-decoder layer.

    Returns the number of modules compiled. State-dict keys are unchanged.
    """
    n = 0
    for blk in [*model.self_layers, *model.cross_layers]:
        for mod in (blk.norm1, blk.mixer, blk.norm2, blk.ffn):
            mod.compile(dynamic=False, **kwargs)
            n += 1
    return n
