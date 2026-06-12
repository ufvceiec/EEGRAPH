# EEGRAPH — Claude Code Guide

## Project overview

EEGraph is an open-source Python library that models electroencephalograms (EEGs) as graphs, enabling brain connectivity analysis. Nodes represent EEG electrodes (up to 333 positions from the 10/20, 10/10, and 10/5 systems). Edges are determined by one of 12 connectivity measures applied to pairs of channels, or by selecting the top-percentage channels for single-channel measures. Published in *Neurocomputing* 519 (2023) 127–134.

**Cite as:**
> Maitin, A.M., Nogales, A., Chazarra, P., & García-Tejedor, Á.J. (2023). EEGraph: An open-source Python library for modeling electroencephalograms using graphs. *Neurocomputing*, 519, 127–134. https://doi.org/10.1016/j.neucom.2022.11.050

---

## Repository layout

```
eegraph/
  __init__.py         — exports Graph (entry point for users)
  graph.py            — Graph class: load_data(), modelate(), visualize_html(), visualize_png()
  importData.py       — InputData class: reads EEG files via MNE; optional electrode montage
  modelateData.py     — ModelData class: orchestrates the connectivity workflow
  strategy.py         — Strategy pattern: one class per connectivity measure + workflow bases
  tools.py            — Pure functions: time intervals, FFT bands, connectivity dispatch,
                        graph construction, Plotly visualisation helpers
  io/
    deap.py           — Helper for the DEAP dataset format
tests/
  tests.py            — unittest suite (30 tests)
  test_files.tar.enc  — encrypted EEG test files (requires key; not in repo)
Examples/
  EEGraph_Example.ipynb
  deap_example.py
```

---

## Architecture — Strategy pattern

`strategy.py` defines abstract/concrete Strategy classes that encode *workflows*. Each concrete estimator inherits from one workflow base:

| Workflow base | When used |
|---|---|
| `Connectivity_No_Bands` | Single scalar per channel pair; no bands |
| `Connectivity_With_Bands` | Per-band scalar per channel pair |
| `Connectivity_single_channel_No_Bands` | Single scalar per channel |
| `Connectivity_single_channel_With_Bands` | Per-band scalar per channel |
| `Cross_correlation_rescaled` | Cross-correlation variants (rescales input first) |
| `Dtf_With_Bands` | DTF via SCoT; directed graphs |

`ModelData` in `modelateData.py` calls `_strategy.calculate_connectivity_workflow()` then `_strategy.make_graph_workflow()`.

`Graph.modelate()` in `graph.py` instantiates the right estimator with `globals()[cls_name]()` (keyed from the `connectivity_measures` dict in `tools.py`).

---

## Adding a new connectivity measure

1. In `strategy.py`: create a class inheriting from the appropriate workflow base, implement `__init__` (set `self.threshold`) and `calculate_conn()` (or `single_channel_conn()`).
2. In `tools.py`: add an entry to `connectivity_measures` dict: `'my_measure': 'My_measure_Estimator'`.
3. Add tests in `tests/tests.py`.

---

## Key data structures

- **`data_intervals`** — 1-D numpy array per row, shape `(channels × intervals,)`, ordered `[ch0_int0, ch1_int0, …, chN_int0, ch0_int1, …]`.
- **`steps`** — list of `(start, end)` sample-index pairs, one per interval.
- **`connectivity_matrix`** — `np.ndarray` shape `(G, N, N)` where `G = intervals` (×bands for band measures), `N = channels`.
- **`flag`** — set to `1` on the strategy when the EEG duration is shorter than one window; the duplicated padding interval is stripped from the output.

---

## Connectivity measures reference

| String key | Class | Domain | Bands required | Default threshold |
|---|---|---|---|---|
| `cross_correlation` | `Cross_correlation_Estimator` | Time | No | 0.5 |
| `pearson_correlation` | `Pearson_correlation_Estimator` | Time | No | 0.7 |
| `squared_coherence` | `Squared_coherence_Estimator` | Frequency | Yes | 0.65 |
| `imag_coherence` | `Imag_coherence_Estimator` | Frequency | Yes | 0.4 |
| `corr_cross_correlation` | `Corr_cross_correlation_Estimator` | Time | No | 0.1 |
| `wpli` | `Wpli_Estimator` | Frequency | Yes | 0.45 |
| `plv` | `Plv_Estimator` | Frequency | Yes | 0.8 |
| `pli` | `Pli_No_Bands_Estimator` | Time/Frequency | No | 0.1 |
| `pli_bands` | `Pli_Bands_Estimator` | Frequency | Yes | 0.1 |
| `dtf` | `Dtf_Estimator` | Frequency | Yes | 0.3 |
| `power_spectrum` | `Power_spectrum_Estimator` | Frequency | Yes | 0.25 (25%) |
| `spectral_entropy` | `Spectral_entropy_Estimator` | Frequency | Yes | 0.25 (25%) |
| `shannon_entropy` | `Shannon_entropy_Estimator` | Time | No | 0.25 (25%) |

Measures with `threshold = 0.25` are single-channel: the threshold is the top-percentage of nodes to connect, not an edge weight cutoff.

---

## Supported input formats

| Extension | MNE reader |
|---|---|
| `.edf` | `mne.io.read_raw_edf` |
| `.gdf` | `mne.io.read_raw_gdf` |
| `.vhdr` | `mne.io.read_raw_brainvision` |
| `.cnt` | `mne.io.read_raw_cnt` |
| `.bdf` | `mne.io.read_raw_bdf` |
| `.egi` | `mne.io.read_raw_egi` |
| `.mff` | `mne.io.read_raw_egi` |
| `.nxe` | `mne.io.read_raw_eximia` |

---

## Running tests

```bash
# All tests (22 unit + 8 integration; integration require decrypted test files)
python -m pytest tests/tests.py -v

# Unit tests only (no EEG files needed)
python -m pytest tests/tests.py -v -k "not (test_load_data or test_modelate or test_visualize)"
```

**Known test failures:**
- `TestImportData`, `TestModelData`, `TestVisualizeData` — require `.test_eeg.gdf` and `.chb02_16.edf` from the encrypted `tests/test_files.tar.enc`.
- `test_calculate_dtf` — `scot 0.2.1` uses `scipy.shape` which was removed in scipy ≥ 1.10. A future `scot` upgrade will fix this.

---

## Known compatibility issues

- **NumPy ≥ 1.25**: assignment of a 1-element array to a scalar position (`arr[0] = Y[mask]`) now requires `Y[mask][0]`. All affected lines in `tools.py` and `strategy.py` have been fixed.
- **SciPy ≥ 1.10 + scot 0.2.1**: `scipy.shape` removed. The DTF measure is broken until scot publishes a compatible release.
- **Python 3.13**: fully compatible after the fixes applied to this branch.

---

## Dependencies

```
numpy==1.22          (tested; ≥1.22 works, warnings appear on ≥1.25 — fixed)
pandas>=1.1.5
mne==1.2             (tested; ≥1.2 works)
networkx>=2.5
plotly>=4.14.3
scipy>=1.1.0
scot==0.2.1          (pinned; DTF broken on scipy ≥ 1.10)
antropy>=0.1.4
kaleido==0.2.1
```

---

## Frequency bands

| Band | Range |
|---|---|
| Delta | 1–4 Hz |
| Theta | 4–8 Hz |
| Alpha | 8–13 Hz |
| Beta | 13–30 Hz |
| Gamma | 30–45 Hz |

Pass as a list of strings: `bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']`.
