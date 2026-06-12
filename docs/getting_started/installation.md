# Installation

## Requirements

EEGraph requires **Python 3.7 or later** and the following libraries:

| Library | Version | Purpose |
|---------|---------|---------|
| NumPy | ≥ 1.22 | FFT, frequency band division, threshold |
| Pandas | ≥ 1.1.5 | DataFrame handling, electrode montage CSV reading |
| MNE | ≥ 1.2 | Reading EEG files in all supported formats |
| NetworkX | ≥ 2.5 | Graph creation, manipulation, metrics |
| Plotly | ≥ 4.14.3 | Interactive HTML graph visualisation |
| SciPy | ≥ 1.1.0 | Pearson correlation, coherence, Hilbert transform, entropy |
| SCoT | 0.2.1 | DTF computation via MVAR model |
| AntroPy | ≥ 0.1.4 | Spectral entropy (Welch method) |
| Kaleido | 0.2.1 | PNG export |

---

## Install from PyPI

The recommended way to install EEGraph is via `pip`:

```bash
pip install EEGRAPH
```

This will automatically install all required dependencies.

---

## Install from source

To install the latest development version directly from GitHub:

```bash
git clone https://github.com/ufvceiec/EEGRAPH.git
cd EEGRAPH
git checkout develop-refactor
pip install -e .
```

---

## Verify installation

Open a Python interpreter and run:

```python
import eegraph
G = eegraph.Graph()
print("EEGraph installed correctly.")
```

---

## Known compatibility notes

!!! warning "SciPy ≥ 1.10 + SCoT 0.2.1"
    The `dtf` connectivity measure relies on SCoT, which uses `scipy.shape` — removed in SciPy 1.10. All other connectivity measures work normally. A compatible SCoT release will resolve this.

!!! info "NumPy ≥ 1.25"
    EEGraph v0.1.17+ is fully compatible with NumPy ≥ 1.25.

!!! info "Python 3.13"
    EEGraph v0.1.17+ is compatible with Python 3.13.
