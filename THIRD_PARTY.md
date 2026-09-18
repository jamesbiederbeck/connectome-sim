# connectome-sim attribution and license scope

This is an independent research project, not affiliated with or endorsed by
FlyEM, Janelia, the MaleCNS collaboration, or the cited researchers.

## Original project material

The [MIT license](LICENSE) covers the original DOOMFLY-derived Python/C++
code inherited via [androsophila](https://github.com/jamesbiederbeck/androsophila),
and the GPU backend added in that fork, under the two separate copyright
notices in LICENSE.

## MaleCNS v1.0 connectome

Credit: the MaleCNS collaboration, including FlyEM at HHMI Janelia, the University
of Cambridge Department of Zoology, the MRC Laboratory of Molecular Biology, and
Google Research, with the authors and contributors identified by the release.

- Dataset and release: https://male-cns.janelia.org/download/
- Paper: https://doi.org/10.1016/j.cell.2026.08.015
- License: [Creative Commons Attribution 4.0 International](licenses/CC-BY-4.0.txt)
  (https://creativecommons.org/licenses/by/4.0/), as linked by the release page.
- Scope: MaleCNS source-derived annotations, connectivity summaries and rendered
  dataset information in `data-provenance/malecns_v1/`, and experiment outputs.
  The large original graph is downloaded separately.
- Changes: entries are normalized, explicit non-neuronal/unresolved objects are
  accounted for, and all released edges among retained neurons are preserved.
  Model weights, sensory mappings and analyses are project transformations; they
  are not measurements supplied or validated by the dataset creators.
- Source URLs, immutable input hashes, upstream filters and exact retention rules
  appear in `datasets.json` and `data-provenance/malecns_v1/`.

## Dopamine and memory source data

Huang, C., Luo, J., Woo, S.J. et al. *Dopamine-mediated interactions between short-
and long-term memory dynamics*. Nature 634, 1141–1149 (2024).
https://doi.org/10.1038/s41586-024-07819-w

The article is [CC BY 4.0](licenses/CC-BY-4.0.txt); separately credited third-party
material may have different terms. `research/huang-2024/targets.json` extracts
selected rates and computes summary statistics from the cited supplementary
workbooks. It retains their hashes and source cells. The adapted model in
`physiology/` is not an author-endorsed reproduction of the complete paper.

The external workbooks and paper PDF are not bundled. See
[retrieval instructions](research/huang-2024/README.md) to obtain them from
the publisher.

## Scientific model references

The LIF framework references Shiu et al. (2024),
https://doi.org/10.1038/s41586-024-07763-9. The upstream model is MIT licensed,
copyright Philip Shiu and Nico Spiller; its original notice is retained in
`licenses/Shiu-model-MIT.txt`. The upstream model checkout is not bundled.
Other scientific papers are cited in the methods and review documents; citing a
paper or implementing an equation does not grant rights to republish its figures.
