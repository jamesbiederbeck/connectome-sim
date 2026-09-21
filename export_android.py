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

ROOT = Path(__file__).absolute().parents[1]

# Exactly the arrays doom/engine.py's Brain.__init__ requires, with the dtype
# it validates each one against. 'superclass' and the provenance arrays stay
# on the desktop: the kernel never reads them.
CONTRACT = [
    ('ptr', np.int64), ('post', np.int32), ('weight', np.float32),
    ('retina', np.int32), ('uv', np.float32),
    ('lamina', np.int32), ('sugar', np.int32),
]


def export(dataset='malecns_v1'):
    source = ROOT / 'outputs/connectome_sim' / dataset / 'graph.npz'
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

    # The readout cells prepare.py already identified, carried across so the
    # device can report their firing without re-deriving cell identity. These are
    # observation points only: nothing downstream of them feeds back into the
    # simulation, and prepare.py's motor_interface caveat travels with them.
    prepared = json.loads((source.parent / 'manifest.json').read_text())
    readouts = [{'index': int(r['index']), 'id': str(r['id']),
                 'type': str(r['type']), 'side': str(r['side'])}
                for r in prepared['readouts']]
    for r in readouts:
        if not 0 <= r['index'] < n:
            raise ValueError(f"readout {r['id']}: graph index out of bounds")

    # Flappy-circuit-specific additions, not part of the core Doom readout set
    # above: the 12-region motor-neuron sprite map (flappy/fly_regions.py) and
    # the haltere afferent clusters a touch pad can stimulate
    # (flappy/haltere_cluster_sweep.py's grouping). Both are read-only index
    # lists into the same graph; see those modules' docstrings for how each
    # region/cluster boundary was derived and what it does and doesn't claim.
    from flappy.fly_regions import REGION_INFO, compute_regions
    from flappy.haltere_cluster_sweep import haltere_clusters
    from flappy.circuit import haltere_afferents
    from connectome_sim.physiology.common import annotations

    class _IdsOnly:
        """haltere_clusters/haltere_afferents only ever read brain.ids -- avoid
        loading the native kernel just to get a graph index/bodyId lookup."""
        def __init__(self, ids): self.ids = ids
    brain = _IdsOnly(a['ids'])
    region_members = compute_regions(a['ids'])
    motor_regions = [{'id': region_id, 'hex': hex_color, 'description': desc,
                       'indices': [int(i) for i in region_members[region_id]]}
                      for region_id, (hex_color, desc) in REGION_INFO.items()]
    for r in motor_regions:
        for i in r['indices']:
            if not 0 <= i < n:
                raise ValueError(f"region {r['id']}: graph index out of bounds")

    clusters = haltere_clusters(brain)
    haltere_idx = set(int(i) for i in haltere_afferents(brain))
    haltere_clusters_out = [{'name': name, 'indices': [int(i) for i in idx]}
                             for name, idx in clusters.items()]
    for c in haltere_clusters_out:
        for i in c['indices']:
            if i not in haltere_idx:
                raise ValueError(f"haltere cluster {c['name']}: index {i} is not a haltere afferent")

    # The two haltere afferents flybody-connectome's single-cell sweep found
    # responsive alone (out of SApp's 148-cell undifferentiated bulk): SApp R
    # body_id 101048 and SApp L body_id 136883 (see connectome-lab/experiments-
    # summary.tsv and flybody-connectome/experiments/LOG.md, 2026-09-19 entry).
    # Exported by body_id, not by cluster grouping, for a device-side lab that
    # drives exactly these two cells independently.
    SAPP_PAIR_BODY_IDS = [('SApp_R', 101048), ('SApp_L', 136883)]
    ids_str = a['ids'].astype(str)
    sapp_pair_out = []
    for name, body_id in SAPP_PAIR_BODY_IDS:
        matches = np.flatnonzero(ids_str == str(body_id))
        if len(matches) != 1:
            raise ValueError(f"expected exactly one graph index for body_id {body_id}, "
                             f"got {len(matches)}")
        idx = int(matches[0])
        if idx not in haltere_idx:
            raise ValueError(f"sapp_pair {name} (body_id {body_id}): index {idx} is not a "
                             f"haltere afferent")
        sapp_pair_out.append({'name': name, 'body_id': str(body_id), 'index': idx})

    # Corazonin (CRZ) neurons: two confidently-typed, bilateral subtypes in
    # this dataset's `type` annotation, CRZ01 and CRZ02 (4 cells total). Each
    # channel pools its L+R pair -- unlike sapp_pair, side is not the
    # scientific question here, subtype identity is. Five further cells carry
    # only the ambiguous compound flywireType label "CRZ01,CRZ02" (tracing
    # didn't resolve which subtype) and are deliberately excluded: only
    # cleanly-typed cells are wired into a device-side stimulation channel.
    # No prior single-cell response sweep exists for these (unlike SApp's
    # 8/14/20mV table) -- this is a first stimulation, not a replication.
    CRZ_TYPES = ['CRZ01', 'CRZ02']
    types_str = annotations(a['ids']).type.astype(str).to_numpy()
    crz_pair_out = []
    for t in CRZ_TYPES:
        matches = np.flatnonzero(types_str == t)
        if len(matches) != 2:
            raise ValueError(f"expected exactly 2 cells (L+R) typed {t}, got {len(matches)}")
        crz_pair_out.append({'name': t, 'body_ids': [str(x) for x in a['ids'][matches]],
                             'indices': [int(i) for i in matches]})

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
        'readouts': readouts,
        'motor_interface': prepared['motor_interface'],
        'motor_regions': motor_regions,
        'haltere_clusters': haltere_clusters_out,
        'sapp_pair': sapp_pair_out,
        'crz_pair': crz_pair_out,
        'note': 'Full retained connectome. No edge cropping, pruning or reordering.',
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    summary = {k: v for k, v in manifest.items()
               if k not in ['blobs', 'readouts', 'motor_regions', 'haltere_clusters', 'sapp_pair', 'crz_pair']}
    summary['readouts'] = len(readouts)
    summary['motor_regions'] = {r['id']: len(r['indices']) for r in motor_regions}
    summary['haltere_clusters'] = {c['name']: len(c['indices']) for c in haltere_clusters_out}
    summary['sapp_pair'] = {c['name']: c['index'] for c in sapp_pair_out}
    summary['crz_pair'] = {c['name']: c['indices'] for c in crz_pair_out}
    print(json.dumps(summary, indent=2))
    return manifest


if __name__ == '__main__':
    export()
