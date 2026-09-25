"""Coverage for Brain.reset()'s generic STATE_FIELDS/_snapshot_state
mechanism, independent of the MemoryBrain-level tests in
test_doom_learning_v6.py. Uses a tiny synthetic 4-cell graph so it needs
neither the real MaleCNS graph nor a GPU -- only the compiled native kernel,
same as the existing MemoryBrain tests."""
import numpy as np
import pytest

from connectome_sim.engine import Brain
from connectome_sim.native import NativeBrain
from connectome_sim.photoreceptor import DARK_SEMISATURATION


def graph(tmp_path):
    """4 cells: 0=retina, 1=lamina, 2=sugar, 3=other, one synapse 0->1."""
    p = tmp_path / 'graph.npz'
    n = 4
    np.savez(p, ptr=np.array([0, 1, 1, 1, 1], dtype=np.int64), post=np.array([1], dtype=np.int32),
              weight=np.array([.5], dtype=np.float32), ids=np.arange(n, dtype=np.int64),
              retina=np.array([0], dtype=np.int32), uv=np.array([[.5, .5]], dtype=np.float32),
              lamina=np.array([1], dtype=np.int32), sugar=np.array([2], dtype=np.int32),
              superclass=np.array(['retina', 'lamina', 'sugar', 'other']))
    return p


def mutate(b):
    b.v[:] = 999; b.g[:] = 1; b.drive[:] = 1; b.refractory[:] = 5
    b.queue[:] = 7; b.queue_count[:] = 3; b.counts[:] = 9
    b.luminance[:] = .5; b.retinal_adaptation[:] = .9
    b.active[:] = 0; b.active_flag[:] = 0; b.nactive[0] = 0
    b.cursor = 50; b.total_spikes = 100; b.sim_ms = 500


def test_brain_reset_restores_mutated_state(tmp_path):
    b = Brain(graph(tmp_path))
    initial = {name: getattr(b, name).copy() for name in b.STATE_FIELDS}
    mutate(b)
    b.reset()
    for name, snapshot in initial.items():
        np.testing.assert_array_equal(getattr(b, name), snapshot, err_msg=name)
    assert b.cursor == 0 and b.total_spikes == 0 and b.sim_ms == 0
    # A plain .copy()-based restore has no hardcoded fallback anymore -- this
    # would silently pass with the wrong value (e.g. 0) if the snapshot were
    # ever taken before retinal_adaptation reached its real starting value.
    assert (b.retinal_adaptation == DARK_SEMISATURATION).all()
    expected_active = np.unique(np.r_[b.retina, b.lamina, b.sugar])
    assert b.nactive[0] == len(expected_active)
    np.testing.assert_array_equal(np.sort(b.active[:b.nactive[0]]), expected_active)


def test_native_brain_reset_restores_subclass_fields(tmp_path):
    b = NativeBrain(graph(tmp_path))
    b.previous_drive[:] = 1; b.last[:] = 77
    b.reset()
    assert not b.previous_drive.any()
    assert (b.last == -1).all()


def test_reset_after_step_matches_a_fresh_instance(tmp_path):
    path = graph(tmp_path)
    b = NativeBrain(path)
    b.step([.8], 50, lamina_bias=12.)
    b.reset()
    fresh = NativeBrain(path)
    for name in b.STATE_FIELDS:
        np.testing.assert_array_equal(getattr(b, name), getattr(fresh, name), err_msg=name)
    assert b.cursor == fresh.cursor == 0
    assert b.total_spikes == fresh.total_spikes == 0
    assert b.sim_ms == fresh.sim_ms == 0


def test_state_fields_composition_is_additive():
    """Each subclass must extend STATE_FIELDS, never shadow it -- otherwise
    reset() silently stops restoring an ancestor's fields."""
    from connectome_sim.physiology.brain import MemoryBrain
    assert set(Brain.STATE_FIELDS) <= set(NativeBrain.STATE_FIELDS)
    assert set(NativeBrain.STATE_FIELDS) <= set(MemoryBrain.STATE_FIELDS)


def test_gpu_brain_reset_restores_mutated_state(tmp_path):
    cp = pytest.importorskip('cupy', reason='GPUBrain requires cupy + a CUDA device')
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip('No CUDA device visible to cupy')
    except Exception:
        pytest.skip('No usable CUDA device for cupy')
    from connectome_sim.gpu import GPUBrain
    b = GPUBrain(graph(tmp_path))
    initial = {name: getattr(b, name).copy() for name in b.STATE_FIELDS}
    b.v[:] = 999; b.g[:] = 1; b.drive[:] = 1; b.refractory[:] = 5
    b._queue[:] = 7; b._queue_count[:] = 3; b._counts[:] = 9
    b.luminance[:] = .5; b.retinal_adaptation[:] = .9
    b.cursor = 50; b.total_spikes = 100; b.sim_ms = 500
    b.reset()
    for name, snapshot in initial.items():
        assert bool((getattr(b, name) == snapshot).all()), name
    assert b.cursor == 0 and b.total_spikes == 0 and b.sim_ms == 0
