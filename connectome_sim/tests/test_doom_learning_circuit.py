"""Unit coverage for the parameterized feedback-circuit selection in
connectome_sim/physiology/circuit.py, independent of the real MaleCNS graph."""
import types
import numpy as np
import pandas as pd
import pytest
from connectome_sim.physiology import circuit as circuit_module
from connectome_sim.physiology.circuit import DEFAULT_SPEC, MAX_DAN_CELLS, identify


def synthetic_brain(monkeypatch, types_by_index):
    """4-cell KC->MBON->DAN toy graph with one existing positive-weight KC->MBON
    edge and one DAN->MBON contact, so identify() has something to select."""
    n = len(types_by_index)
    brain = types.SimpleNamespace(
        ids=np.arange(n, dtype=np.int64), n=n,
        ptr=np.array([0, 1, 1, 2, 2], dtype=np.int64),  # cell0(KC)->post[0], cell2(DAN)->post[1]
        post=np.array([1, 1], dtype=np.int32),
        weight=np.array([20., .5], dtype=np.float32))
    frame = pd.DataFrame({'type': types_by_index,
                           'instance': ['x'] * n, 'somaSide': ['L'] * n}, index=brain.ids)
    monkeypatch.setattr(circuit_module, 'annotations', lambda ids: frame.loc[ids])
    return brain


def test_default_spec_matches_kc_mbon11_ppl101(monkeypatch):
    brain = synthetic_brain(monkeypatch, ['KCg-d', 'MBON11', 'PPL101', 'other'])
    result = identify(brain)
    assert result['report']['spec'] == DEFAULT_SPEC
    np.testing.assert_array_equal(result['kc'], [0])
    np.testing.assert_array_equal(result['mb'], [1])
    np.testing.assert_array_equal(result['dan'], [2])
    np.testing.assert_array_equal(result['edges'], [0])


def test_custom_spec_selects_different_cell_types(monkeypatch):
    brain = synthetic_brain(monkeypatch, ['KCg-d', 'MBON01', 'PAM01', 'other'])
    spec = {'kc_prefix': 'KC', 'mbon_types': ['MBON01'], 'dan_types': ['PAM01']}
    result = identify(brain, spec=spec)
    assert result['report']['spec'] == spec
    np.testing.assert_array_equal(result['mb'], [1])
    np.testing.assert_array_equal(result['dan'], [2])


def test_missing_mbon_type_raises(monkeypatch):
    brain = synthetic_brain(monkeypatch, ['KCg-d', 'MBON11', 'PPL101', 'other'])
    with pytest.raises(ValueError, match='MBON'):
        identify(brain, spec={'kc_prefix': 'KC', 'mbon_types': ['NOPE'], 'dan_types': ['PPL101']})


def test_missing_dan_type_raises(monkeypatch):
    brain = synthetic_brain(monkeypatch, ['KCg-d', 'MBON11', 'PPL101', 'other'])
    with pytest.raises(ValueError, match='DAN'):
        identify(brain, spec={'kc_prefix': 'KC', 'mbon_types': ['MBON11'], 'dan_types': ['NOPE']})


def test_too_many_dan_cells_raises_before_kernel_overflow(monkeypatch):
    types_by_index = ['KCg-d', 'MBON11'] + ['PPL101'] * (MAX_DAN_CELLS + 1)
    brain = synthetic_brain(monkeypatch, types_by_index)
    brain.ptr = np.zeros(len(types_by_index) + 1, dtype=np.int64)
    brain.ptr[1] = 1
    brain.post = np.array([1], dtype=np.int32)
    brain.weight = np.array([20.], dtype=np.float32)
    with pytest.raises(ValueError, match='int8 kernel index limit'):
        identify(brain, spec={'kc_prefix': 'KC', 'mbon_types': ['MBON11'], 'dan_types': ['PPL101']})
