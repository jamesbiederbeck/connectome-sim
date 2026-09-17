"""Independent physiological targets; no Doom performance is involved."""
from doom_learning.circuit import DEFAULT_SPEC
from .visual import VisualMemoryBrain

# These currents/rates are Huang et al. 2024 Fig 1d/e baselines measured
# specifically for MBON11 and PPL101 -- they do not transfer to a different
# circuit_spec's cell types, so calibrated_brain refuses anything but the
# default spec rather than silently mislabeling arbitrary cells as calibrated.
_CALIBRATED_SPEC=DEFAULT_SPEC


def calibrated_brain(eta=.001,circuit_spec=None):
    if circuit_spec is not None and circuit_spec!=_CALIBRATED_SPEC:
        raise ValueError('This calibration (Huang et al. 2024 Fig 1d/e) is specific to '
            f'{_CALIBRATED_SPEC!r}; it does not apply to circuit_spec={circuit_spec!r}. '
            'Construct VisualMemoryBrain directly and calibrate that circuit independently.')
    b=VisualMemoryBrain(eta=eta,circuit_spec=circuit_spec)
    b.tonic[b.circuit['mb']]=9.87
    # Best *observed* point in the recorded current sweep, not a bisection
    # interpolation: this recurrent spiking system is not monotonic in bias.
    b.tonic[b.circuit['dan']]=11.3125
    b.dan_baseline_hz[:]=20.09
    b.calibration={'MBON_current':9.87,'DAN_current':11.3125,
        'observed_MBON_hz':[37,37],'observed_DAN_hz':[23,19],
        'target_MBON_hz':37.16625,'target_MBON_sd':9.06014,
        'target_DAN_hz':20.09,'target_DAN_sd':4.30193,
        'evidence':'https://doi.org/10.1038/s41586-024-07819-w',
        'source_data':'Figure 1, panels d/e, n=20 flies per type',
        'limits':'A fitted background current is not identified pacemaker physiology. This only constrains baseline rates, not cue responses, burst statistics or learned behavior.'}
    return b
