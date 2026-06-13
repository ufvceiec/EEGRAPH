<p align="center">
  <img src="https://github.com/ufvceiec/EEGRAPH/blob/develop-refactor/demo/eegraph-logo.png" alt="EEGraph Logo"/>
</p>

[![GP3 License](https://img.shields.io/github/license/ufvceiec/EEGRAPH.svg)](https://github.com/ufvceiec/EEGRAPH/blob/develop-refactor/LICENSE)
[![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/EEGRAPH?color=blue)](https://pypi.org/project/EEGRAPH/)
[![Build Status](https://travis-ci.com/ufvceiec/EEGRAPH.svg?branch=develop-refactor)](https://travis-ci.com/ufvceiec/EEGRAPH)
[![codecov](https://codecov.io/gh/ufvceiec/EEGRAPH/branch/develop-refactor/graph/badge.svg?token=WxnBb2CTTL)](https://codecov.io/gh/ufvceiec/EEGRAPH)

EEGraph is a Python library to model electroencephalograms (EEGs) as graphs, enabling the analysis of brain connectivity between different brain areas. It has applications in the study of neurological diseases like Parkinson's or epilepsy. The graph can be exported as a NetworkX graph-like object or graphically visualized as an interactive HTML plot.

## Citation

If you use this library, please cite:

> Maitin, A. M., Nogales, A., Chazarra, P., & García-Tejedor, Á. J. (2023). EEGraph: An open-source Python library for modeling electroencephalograms using graphs. *Neurocomputing*, 519, 127–134. https://doi.org/10.1016/j.neucom.2022.11.050

## Getting Started

### Dependencies

| Library | Purpose |
|---------|---------|
| NumPy | FFT, frequency band division, threshold |
| Pandas | DataFrame handling, electrode montage CSV reading |
| MNE | Reading EEG files in all supported formats |
| NetworkX | Graph creation, manipulation, adjacency matrix, graph metrics |
| Plotly | Interactive HTML graph visualisation |
| SciPy | Pearson correlation, cross-correlation, coherence, CSD, Hilbert transform, Shannon entropy |
| SCoT | DTF computation (MVAR model) |
| AntroPy | Spectral entropy (Welch method) |
| Kaleido | PNG export |

### Installing EEGraph

```bash
pip install EEGRAPH
```

## Functions

### Documentation
[EEGraph documentation](https://ufvceiec.github.io/EEGRAPH/) is available online. [Examples](https://github.com/ufvceiec/EEGRAPH/blob/develop-refactor/Examples/EEGraph_Example.ipynb) of usage are also available.

### Importing EEG data

| Format | Extension |
|--------|-----------|
| Brainvision | `.vhdr` |
| Neuroscan CNT | `.cnt` |
| European data format | `.edf` |
| Biosemi data format | `.bdf` |
| General data format | `.gdf` |
| EGI simple binary | `.egi` |
| EGI MFF format | `.mff` |
| eXimia | `.nxe` |

### Connectivity Measures

| Measure | Key | Domain | Bands |
|---------|-----|--------|-------|
| Cross Correlation | `cross_correlation` | Time | No |
| Pearson Correlation | `pearson_correlation` | Time | No |
| Corrected Cross Correlation | `corr_cross_correlation` | Time | No |
| Phase Lag Index | `pli` | Time | No |
| Shannon Entropy | `shannon_entropy` | Time | No |
| Squared Coherence | `squared_coherence` | Frequency | Yes |
| Imaginary Coherence | `imag_coherence` | Frequency | Yes |
| Phase Locking Value (PLV) | `plv` | Frequency | Yes |
| Phase Lag Index (bands) | `pli_bands` | Frequency | Yes |
| Weighted Phase Lag Index (WPLI) | `wpli` | Frequency | Yes |
| Directed Transfer Function (DTF) | `dtf` | Frequency | Yes |
| Power Spectrum | `power_spectrum` | Frequency | Yes |
| Spectral Entropy | `spectral_entropy` | Frequency | Yes |

## Usage

### Load data
```python
import eegraph
G = eegraph.Graph()
G.load_data(path='eeg_sample.edf', exclude=['EEG EKG1-EKG2'])
```

#### With electrode montage
```python
G.load_data(path='eeg_sample.gdf', electrode_montage_path='electrodemontage.set.ced')
```

The montage file must contain at least two columns (electrode number and label), separated by whitespace, `;`, or `:`.

### Modelate data

#### Without frequency bands
```python
graphs, connectivity_matrix = G.modelate(window_size=2, connectivity='pearson_correlation')
```

#### With frequency bands
```python
graphs, connectivity_matrix = G.modelate(
    window_size=2,
    connectivity='squared_coherence',
    bands=['delta', 'theta', 'alpha']
)
```

#### Custom threshold
```python
graphs, connectivity_matrix = G.modelate(
    window_size=2,
    connectivity='pearson_correlation',
    threshold=0.8
)
```

### Window size

- **`int`**: uniform window length in seconds — `window_size=2` splits the EEG into 2-second segments.
- **`list`**: explicit interval boundaries in seconds — `window_size=[0, 3, 8]` creates two intervals: [0–3 s] and [3–8 s]. The first value must be `0`.

### Visualize graph
```python
G.visualize_html(graphs[0], 'graph_1')   # saves graph_1_plot.html and opens in browser
G.visualize_png(graphs[0], 'graph_1')    # saves graph_1.png
```

Channel names must be in one of these formats:
- Standard: `Fp1`, `Fp2`, `C3`, `Cz`
- Dash-separated: `Fp1-EEG` (electrode name on the left)
- Space-separated: `EEG Fp1` (electrode name on the right)

### Output

- **`graphs`** — `dict` of NetworkX `Graph` (or `DiGraph` for DTF) objects, one per time window (×bands for band measures).
- **`connectivity_matrix`** — `np.ndarray` of shape `(G, N, N)`, all connectivity values regardless of threshold.

```python
# Save matrix to CSV
import numpy as np
np.savetxt('connectivity.csv', connectivity_matrix[0], delimiter=',')
```

![Connectivity Graph Output Example](https://github.com/ufvceiec/EEGRAPH/blob/develop/demo/eegraph_output.gif)

### Execution example video

https://user-images.githubusercontent.com/41289779/201069364-95fa82dd-a96b-454f-af79-7f124edb7d03.mp4

## EEGraph Workflow
![EEGraph Workflow](https://github.com/ufvceiec/EEGRAPH/blob/develop-refactor/demo/eegraph_workflow.png)

## Frequency Bands

| Band | Range |
|------|-------|
| Delta | 1–4 Hz |
| Theta | 4–8 Hz |
| Alpha | 8–13 Hz |
| Beta | 13–30 Hz |
| Gamma | 30–45 Hz |

## Contributing
See [Contribution guidelines](https://github.com/ufvceiec/EEGRAPH/blob/develop-refactor/CONTRIBUTING.md) for more information.

## Versioning
See [CHANGELOG.txt](CHANGELOG.txt) for major/breaking updates and version history.

## Known Issues

- **DTF + SciPy ≥ 1.10**: `scot 0.2.1` relies on `scipy.shape` which was removed. DTF connectivity is unavailable until a compatible `scot` release.
- **NumPy ≥ 1.25 deprecation**: array-to-scalar assignments have been updated in this branch.

## Contact

Centro de Estudios e Innovación en Gestión del Conocimiento (CEIEC), Universidad Francisco de Vitoria.

- Responsible: Ana María Maitín (a.maitin@ceiec.es), Alberto Nogales (alberto.nogales@ceiec.es)
- Main developer: Pedro Chazarra
- Contributor: Fernando Pérez ([@FernandoPerezLara](https://github.com/FernandoPerezLara))

## License

This project is licensed under the [GPL-3.0 License](https://github.com/ufvceiec/EEGRAPH/blob/develop-refactor/LICENSE).
