"""Exact cell identities and existing connections for a candidate memory rule."""
import numpy as np
from .common import annotations, digest

# The feedback circuit this rule applies to: which existing cell types act as
# the plastic KC presynaptic pool, the MBON postsynaptic pool, and the DAN
# modulatory pool. This is the only place those choices are made; everything
# downstream (brain.py, kernel.cpp, calibration.py) just consumes indices.
# kc_prefix is a startswith() match (KC subtypes: KCg-d, KCab, ...); mbon_types
# and dan_types are exact-match lists, matching the original hardcoded
# behavior this default spec reproduces byte-for-byte.
DEFAULT_SPEC={'kc_prefix':'KC','mbon_types':['MBON11'],'dan_types':['PPL101']}

# int8 in kernel.cpp's dan_index array; a spec matching more DAN cells than
# this would silently wrap into garbage kernel indices instead of failing loudly.
MAX_DAN_CELLS=127


def identify(brain,*,spec=None):
    spec=DEFAULT_SPEC if spec is None else spec
    a=annotations(brain.ids); types=a.type.fillna('')
    kc=np.flatnonzero(types.str.startswith(spec['kc_prefix'])).astype(np.int32)
    mb=np.flatnonzero(types.isin(spec['mbon_types'])).astype(np.int32)
    dan=np.flatnonzero(types.isin(spec['dan_types'])).astype(np.int32)
    if not len(mb):raise ValueError(f'No neurons found matching MBON type(s) {spec["mbon_types"]!r}')
    if not len(dan):raise ValueError(f'No neurons found matching DAN type(s) {spec["dan_types"]!r}')
    if len(dan)>MAX_DAN_CELLS:raise ValueError(f'{len(dan)} DAN cells exceed the int8 kernel index limit ({MAX_DAN_CELLS}); widen dan_index in kernel.cpp before using this spec')
    edges=np.flatnonzero(np.isin(brain.post,mb)).astype(np.int64)
    pre=np.searchsorted(brain.ptr,edges,side='right')-1
    keep=np.isin(pre,kc);edges=edges[keep];pre=pre[keep].astype(np.int32)
    if not len(edges):raise ValueError(f'No existing edges from KC ({spec["kc_prefix"]}*) to MBON {spec["mbon_types"]!r} found')
    if np.any(brain.weight[edges]<=0):raise ValueError('Candidate KC-to-MBON inputs must be existing positive-weight edges.')
    # Anatomical pooling proxy. Direct DAN->MBON contact weights specify
    # fractional participation, not measured dopamine release or receptor gain.
    contact=np.zeros((len(dan),len(mb)),dtype=np.float32)
    for d,i in enumerate(dan):
        for m,j in enumerate(mb):
            ix=np.flatnonzero(brain.post[brain.ptr[i]:brain.ptr[i+1]]==j)+brain.ptr[i]
            contact[d,m]=np.abs(brain.weight[ix]).sum()
    if np.any(contact.sum(axis=0)<=0):raise ValueError('Missing reconstructed DAN-to-MBON contact support.')
    gain=(contact/contact.sum(axis=0))[...,np.searchsorted(mb,brain.post[edges])].copy()
    mask=np.zeros(brain.n,dtype=np.uint8);mask[kc]=1
    dan_index=np.full(brain.n,-1,dtype=np.int8);dan_index[dan]=np.arange(len(dan))
    def cells(indices):return [{'index':int(i),'id':str(brain.ids[i]),'type':str(a.type.iloc[i]),
                               'instance':str(a.instance.iloc[i]),'soma_side':str(a.somaSide.iloc[i])} for i in indices]
    report={'dataset':'MaleCNS v1.0','neurons':brain.n,'edges':len(brain.weight),'kc_count':len(kc),
        'spec':spec,'plastic_edges':len(edges),'plastic_edge_index_sha256':digest(edges),
        'plastic_presynaptic_index_sha256':digest(pre),'DAN':cells(dan),'MBON':cells(mb),
        'gamma_d_KC':cells(np.flatnonzero(types.eq('KCg-d'))),
        'dan_to_mbon_contact_fraction':(contact/contact.sum(axis=0)).tolist(),
        'selection':f'All existing KC ({spec["kc_prefix"]}*) to MBON {spec["mbon_types"]!r} edges; no graph pruning or added connections.',
        'compartment_assignment':'Cell-type-level gamma1/peduncle inference; aggregate edges do not resolve receptor localization.',
        'modulation':f'{spec["dan_types"]!r} outgoing edges deliver a transmitter trace to every recorded target. Only the specified KC-to-MBON efficacy response is modeled. Other target responses remain unknown.',
        'baseline_difference':f'{spec["dan_types"]!r} cells use a modulatory channel rather than the baseline default fast excitatory sign.',
        'evidence':['https://doi.org/10.1016/j.neuron.2015.11.003','https://doi.org/10.7554/eLife.16135','https://male-cns.janelia.org/download/']}
    return {'kc':kc,'mb':mb,'dan':dan,'edges':edges,'pre':pre,'gain':gain.astype(np.float32),
            'kc_mask':mask,'dan_index':dan_index,'report':report}
