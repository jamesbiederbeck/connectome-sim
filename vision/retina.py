"""Pluggable photoreceptor-sampling interface, factored out of doom/game.py so a
second game harness (flappy/) can reuse it without depending on doom's ViZDoom
boundary, and so richer models can be swapped in later without touching either
game harness.

The only implementation here today is a thin wrapper around
connectome_sim.photoreceptor.retinal_samples -- the exact, already-validated
bilinear-sample-plus-sRGB-to-linear-luma transform, now factored into its own
vizdoom-free module. doom/game.py re-exports it unchanged for existing
callers. Nothing about Doom's numeric behavior is touched by this module.

The interface exists for future work, not implemented here: a richer
per-receptor model along the lines of ~/code/android/bugvision's drosophila
compound-eye prototype (CompoundEye.java's hex-lattice retinotopic projection
with an acceptance-angle box blur per ommatidium, and stages.py's Naka-Rushton
light-adaptation stage), and eventually a synthetic/projected light source
composited onto a rendered frame before sampling. None of that is ported here;
this module only gives it a place to plug in as a second PhotoreceptorModel
subclass, selected by whichever game harness wants it, without changing the
harnesses' call sites.
"""


class PhotoreceptorModel:
    def sample(self, rgb, uv):
        """rgb: (H,W,3) uint8 array. uv: (N,2) float array in [0,1]. Returns
        float32[N] luminance, one value per receptor in uv's order."""
        raise NotImplementedError


class BilinearLuminance(PhotoreceptorModel):
    """Current behavior, unchanged. Wraps connectome_sim.photoreceptor.retinal_samples verbatim."""

    def sample(self, rgb, uv):
        from connectome_sim.photoreceptor import retinal_samples
        return retinal_samples(rgb, uv)
