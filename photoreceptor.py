"""Pure photoreceptor sampling math -- no ViZDoom, no game state. Factored out
of doom/game.py so it can move into the connectome-engine split without
dragging the vizdoom import along with it. doom/game.py and vision/retina.py
both import retinal_samples from here; its numeric behavior is unchanged.
"""
import numpy as np


def retinal_samples(rgb, uv):
    """Bilinear luminance at receptor samples only. No scene interpretation."""
    h, w = rgb.shape[:2]
    x = uv[:, 0] * (w - 1)
    y = uv[:, 1] * (h - 1)
    x0 = x.astype(int)
    y0 = y.astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    dx = x - x0
    dy = y - y0

    def linear_luma(pixels):
        p = pixels.astype(np.float32) / 255
        p = np.where(p <= .04045, p / 12.92, ((p + .055) / 1.055) ** 2.4)
        return p @ np.asarray([.2126, .7152, .0722], dtype=np.float32)

    return ((1 - dx) * (1 - dy) * linear_luma(rgb[y0, x0])
             + dx * (1 - dy) * linear_luma(rgb[y0, x1])
             + (1 - dx) * dy * linear_luma(rgb[y1, x0])
             + dx * dy * linear_luma(rgb[y1, x1])).astype(np.float32)


# --- Photoreceptor drive -------------------------------------------------
# Luminance reaches the network as an injected current through a Naka-Rushton
# curve, gain * L / (K + L). K is the semisaturation: the luminance at which
# the receptor gives half its maximum current, and the point where the curve is
# steepest and the receptor most sensitive.
#
# K was a fixed 0.02 everywhere. Measured against a lit MuJoCo scene whose
# median receptor luminance is 0.53, that puts every receptor 27x past
# semisaturation, on the flat top of the curve: sensitivity there is 1.96 mV
# per unit luminance against 375 at K, so the eye ran 192x less sensitive than
# the same curve allows. A fly swatter filling the dorsal field dropped mean
# luminance 44% and mean drive 2.8%.
#
# Real photoreceptors do not have a fixed semisaturation. They adapt, sliding
# K onto the ambient level so the response stays centred wherever the overall
# light level sits. That is what this adds: K follows a slow per-receptor mean
# of luminance, floored at the original 0.02 so darkness cannot divide it away.
# With K at ambient, a receptor sitting at the mean gives half its maximum
# current and the steepest available response to change.
#
# `tau_ms=None` restores the fixed constant exactly, for comparison with any
# result recorded before this existed.
RETINAL_GAIN = 30.
DARK_SEMISATURATION = .02
ADAPTATION_MS = 500.


def adapted_drive(luminance, adaptation, elapsed_ms, *, tau_ms=ADAPTATION_MS,
                  floor=DARK_SEMISATURATION, gain=RETINAL_GAIN):
    """Injected current for each receptor. Updates `adaptation` in place.

    `adaptation` is the per-receptor running mean of luminance and carries the
    operating point between calls; `luminance` is the already-filtered signal.
    A long `tau_ms` relative to the frame interval is the point -- adaptation
    has to be slow enough that a passing object moves the response rather than
    being absorbed by the operating point tracking it.
    """
    import math
    if tau_ms is None:
        return gain * luminance / (floor + luminance)
    if not (math.isfinite(tau_ms) and tau_ms > 0):
        raise ValueError('Positive adaptation time constant required')
    adaptation += (1 - math.exp(-elapsed_ms / tau_ms)) * (luminance - adaptation)
    return gain * luminance / (np.maximum(adaptation, floor) + luminance)
