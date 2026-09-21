"""Bin per-cell spike counts into named populations, for any caller.

Written because three separate ad hoc scripts (SApp/CRZ latch characterization,
a leg-loop run) each reimplemented the same handful of lines: look up an
annotation column, group counts by it, report cells/total_spikes/mean_hz. One
of those scripts reached into `flappy.fly_regions` for a body-region grouping,
which pulls a downstream consumer's package into what should be a generic
engine utility -- connectome_sim has no business importing flappy, and
androsophila/flybody-connectome callers have no business reimplementing this
themselves either. This module owns exactly the counts-to-populations binning;
it never chooses what the populations *are*.

Two ways to name a population, and they compose in one call:

- `by`: an `annotations()` column name (default 'superclass') -- every value
  that column takes becomes one population automatically. This is the only
  thing connectome_sim itself knows how to group by, since it's the one
  taxonomy this engine's own annotation data provides.
- `extra`: named boolean masks the *caller* supplies for whatever grouping is
  specific to its own domain -- DLM motor neurons, a fly-body region, a
  haltere cluster, the two cells of an experiment's own stimulation channel.
  Keeps that domain knowledge in the repo that owns it (flappy-haltere's
  `fly_regions`/`circuit`, flybody-connectome's `fly.legs`, ...), not here.
"""
from __future__ import annotations

import numpy as np

from connectome_sim.physiology.common import annotations


def population_report(ids, counts, duration_ms, *, by: str = 'superclass',
                      extra: dict | None = None) -> dict:
    """One row per population: cell count, total spikes, mean Hz/cell.

    Args:
        ids: per-cell body IDs, in the same order as `counts` (`Brain.ids`).
        counts: per-cell spike counts over the window being reported (a
            `Brain.step`/`GPUBrain.step` return value, or an accumulated sum
            of several such calls -- this function doesn't care which, only
            that `counts` and `duration_ms` describe the same window).
        duration_ms: simulated milliseconds `counts` was accumulated over,
            for the Hz conversion. Get this wrong (e.g. pass a per-step
            duration against multi-step accumulated counts) and every rate
            here is wrong by that same factor -- nothing here can detect
            that mismatch, so callers own getting it right.
        by: an `annotations()` column name; every distinct value present
            becomes one population. Cells with a missing/NaN value in this
            column are silently excluded from `by`-populations (they still
            count toward any `extra` mask that includes them).
        extra: optional `{name: boolean_mask}` for caller-specific groupings
            layered on top of `by` in the same report -- see module
            docstring for why this exists instead of `by` supporting
            arbitrary domain taxonomies.

    Returns:
        `{population_name: {"cells": int, "total_spikes": int,
        "mean_hz_per_cell": float}}`, plus a "TOTAL" row over every cell in
        `counts` regardless of `by`/`extra`.
    """
    counts = np.asarray(counts)
    if counts.shape != (len(ids),):
        raise ValueError(f"counts shape {counts.shape} does not match {len(ids)} ids")
    if not np.isfinite(duration_ms) or duration_ms <= 0:
        raise ValueError("duration_ms must be finite and positive")
    rate_hz = counts.astype(np.float64) / (duration_ms / 1000.0)

    def row(mask):
        n = int(mask.sum())
        if n == 0:
            return {"cells": 0, "total_spikes": 0, "mean_hz_per_cell": 0.0}
        return {"cells": n, "total_spikes": int(counts[mask].sum()),
                "mean_hz_per_cell": round(float(rate_hz[mask].mean()), 3)}

    a = annotations(ids)
    if by not in a.columns:
        raise ValueError(f"'{by}' is not an annotations() column")
    # fillna before astype(str): a missing value in an object-dtype column
    # can survive .astype(str) as an actual float NaN rather than becoming
    # the string "nan" (version-dependent pandas behavior), which then
    # breaks np.unique's sort by mixing float and str in one array.
    missing = "__missing__"
    labels = a[by].fillna(missing).astype(str).to_numpy()

    report = {}
    for label in np.unique(labels):
        if label == missing:
            continue
        report[label] = row(labels == label)
    for name, mask in (extra or {}).items():
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(ids),):
            raise ValueError(f"extra[{name!r}] mask shape {mask.shape} does not match {len(ids)} ids")
        report[name] = row(mask)
    report["TOTAL"] = row(np.ones(len(ids), dtype=bool))
    return report
