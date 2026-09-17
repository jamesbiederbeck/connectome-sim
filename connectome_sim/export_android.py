"""Emit the prepared graph as raw little-endian blobs an Android build can mmap.

The device has no Arrow, no numpy and no Python. Rather than teach it to read
graph.npz, this writes each array the native kernel needs as a flat,
C-contiguous, little-endian file plus a manifest recording n, the edge count,
every dtype and a sha256 per blob. The Android loader mmaps the blobs and
checks those digests, so a run on the phone is traceable to the same prepared
graph as a run on the desktop -- the sidecar discipline doom/native.py applies
to the kernel binary, applied to the data.

Every released edge is exported. Nothing is subset, pruned or reordered.
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Exactly the arrays doom/engine.py's Brain.__init__ requires, with the dtype
# it validates each one against. 'superclass' and the provenance arrays stay
# on the desktop: the kernel never reads them.
CONTRACT = [
    ('ptr', np.int64), ('post', np.int32), ('weight', np.float32),
    ('retina', np.int32), ('uv', np.float32),
    ('lamina', np.int32), ('sugar', np.int32),
]


def export(dataset='malecns_v1'):
    source = ROOT / 'outputs/doom' / dataset / 'graph.npz'
    out = source.parent / 'android'
    out.mkdir(parents=True, exist_ok=True)
    a = np.load(source)
    n = len(a['ids'])
    blobs = {}
    for key, dtype in CONTRACT:
        x = a[key]
        if x.dtype != dtype:
            raise ValueError(f'{key}: expected {dtype}, prepared graph has {x.dtype}')
        x = np.ascontiguousarray(x, dtype=dtype)
        if sys.byteorder != 'little':
            x = x.byteswap()
        path = out / f'{key}.bin'
        path.write_bytes(x.tobytes(order='C'))
        blobs[key] = {'dtype': np.dtype(dtype).name, 'shape': list(x.shape),
                      'bytes': path.stat().st_size,
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    # Same structural invariants engine.py enforces on load, checked here so a
    # malformed export is caught on the desktop rather than on the phone.
    ptr = a['ptr']
    if ptr.shape != (n + 1,) or ptr[0] != 0 or ptr[-1] != len(a['post']):
        raise ValueError('Invalid CSR graph')
    if len(a['weight']) != len(a['post']) or not np.isfinite(a['weight']).all():
        raise ValueError('Invalid synaptic weights')
    for key in ['post', 'retina', 'lamina', 'sugar']:
        if np.any(a[key] < 0) or np.any(a[key] >= n):
            raise ValueError(f'{key}: graph index out of bounds')

    manifest = {
        'dataset': dataset,
        'neurons': int(n),
        'edges': int(len(a['post'])),
        'retina': int(len(a['retina'])),
        'lamina': int(len(a['lamina'])),
        'sugar': int(len(a['sugar'])),
        'byteorder': 'little',
        'source_npz_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'blobs': blobs,
        'note': 'Full retained connectome. No edge cropping, pruning or reordering.',
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'blobs'}, indent=2))
    return manifest


if __name__ == '__main__':
    export()
