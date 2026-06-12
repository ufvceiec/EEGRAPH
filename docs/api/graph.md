# API Reference — `Graph`

The `Graph` class is the primary entry point for EEGraph. Import it with:

```python
import eegraph
G = eegraph.Graph()
```

All public methods are documented below.

---

## `Graph.load_data(path, exclude=[], electrode_montage_path=None)`

Load an EEG recording from disk into the `Graph` object.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | — | Path to the EEG file. Supported extensions: `.edf`, `.gdf`, `.vhdr`, `.cnt`, `.bdf`, `.egi`, `.mff`, `.nxe`. |
| `exclude` | `list[str]` | `[]` | Channel names to drop before processing (e.g. `['EEG EKG1-EKG2']`). |
| `electrode_montage_path` | `str \| None` | `None` | Path to a custom electrode montage file (`.ced`, `.tsv`, or whitespace/`;`/`:`-delimited text). Overrides standard channel names for visualization. |

**Returns** — `None`. The loaded data is stored internally.

**Raises**

- `ValueError` — if the file extension is not supported.
- `FileNotFoundError` — if `path` does not exist.

**Example**

```python
G = eegraph.Graph()

# Basic load
G.load_data(path='eeg_sample.edf')

# Exclude non-EEG channels
G.load_data(path='eeg_sample.edf', exclude=['EEG EKG1-EKG2', 'EEG EOG'])

# Custom electrode montage
G.load_data(
    path='eeg_sample.gdf',
    electrode_montage_path='electrodemontage.set.ced'
)
```

---

## `Graph.modelate(window_size, connectivity, bands=None, threshold=None)`

Compute brain connectivity and return graphs for each time window (and band, if applicable).

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_size` | `int \| list[int]` | — | Window length in seconds (int), or explicit boundary list starting with `0` (list). See [Window Size & Epochs](../user_guide/windowing.md). |
| `connectivity` | `str` | — | Connectivity measure key (see table below). |
| `bands` | `list[str] \| None` | `None` | Frequency bands for band-based measures. Valid values: `'delta'`, `'theta'`, `'alpha'`, `'beta'`, `'gamma'`. Required for frequency-domain measures. |
| `threshold` | `float \| None` | `None` | Edge inclusion threshold. If `None`, the measure's built-in default is used. For single-channel measures (`power_spectrum`, `spectral_entropy`, `shannon_entropy`) this is the top-percentage of nodes to connect (0–1). |

**Connectivity measure keys**

| Key | Domain | Bands required | Default threshold |
|-----|--------|----------------|-------------------|
| `cross_correlation` | Time | No | 0.5 |
| `pearson_correlation` | Time | No | 0.7 |
| `corr_cross_correlation` | Time | No | 0.1 |
| `pli` | Time | No | 0.1 |
| `shannon_entropy` | Time | No | 0.25 (top 25%) |
| `squared_coherence` | Frequency | Yes | 0.65 |
| `imag_coherence` | Frequency | Yes | 0.4 |
| `plv` | Frequency | Yes | 0.8 |
| `pli_bands` | Frequency | Yes | 0.1 |
| `wpli` | Frequency | Yes | 0.45 |
| `dtf` | Frequency | Yes | 0.3 |
| `power_spectrum` | Frequency | Yes | 0.25 (top 25%) |
| `spectral_entropy` | Frequency | Yes | 0.25 (top 25%) |

**Returns** — `tuple(graphs, connectivity_matrix)`

- `graphs` — `dict` of NetworkX `Graph` (or `DiGraph` for DTF) objects. Keys are integers for time-domain, or strings like `'0_alpha'` for frequency-domain.
- `connectivity_matrix` — `np.ndarray` of shape `(num_graphs, num_channels, num_channels)` containing all pairwise values before thresholding.

**Example**

```python
# Time-domain, no bands
graphs, matrix = G.modelate(window_size=2, connectivity='pearson_correlation')

# Frequency-domain with bands
graphs, matrix = G.modelate(
    window_size=2,
    connectivity='squared_coherence',
    bands=['alpha', 'beta', 'gamma']
)

# Custom threshold
graphs, matrix = G.modelate(
    window_size=5,
    connectivity='pearson_correlation',
    threshold=0.9
)
```

---

## `Graph.compute_metrics(graphs)`

Compute a set of NetworkX graph-theoretic metrics for every graph in `graphs`.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `graphs` | `dict` | The `graphs` dict returned by `modelate()`. |

**Returns** — `dict` mapping each graph key to a `dict` of metric name → value. Metrics computed include (but are not limited to): degree, clustering coefficient, betweenness centrality, closeness centrality, average shortest path length, and graph density.

**Example**

```python
graphs, matrix = G.modelate(window_size=2, connectivity='pearson_correlation')
metrics = G.compute_metrics(graphs)

for key, m in metrics.items():
    print(f"Window {key}: density={m['density']:.3f}, "
          f"avg_clustering={m['average_clustering']:.3f}")
```

---

## `Graph.visualize_html(graph, name)`

Render a connectivity graph as an interactive HTML file and open it in the default browser.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `graph` | `networkx.Graph` | A single graph from the `graphs` dict. |
| `name` | `str` | Base filename (without extension). The output file is `<name>_plot.html`. |

**Returns** — `None`. Saves `<name>_plot.html` to the current directory and opens it.

**Notes**

- Channel names must be in a recognized format: standard (`Fp1`), dash-separated (`Fp1-EEG`), or space-separated (`EEG Fp1`).
- Channels not matching a standard 10/20, 10/10, or 10/5 electrode position are omitted from the 3-D head visualization with a warning.

**Example**

```python
G.visualize_html(graphs[0], 'window_0_pearson')
# Saves: window_0_pearson_plot.html
```

---

## `Graph.visualize_png(graph, name)`

Render a connectivity graph as a static PNG image.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `graph` | `networkx.Graph` | A single graph from the `graphs` dict. |
| `name` | `str` | Base filename (without extension). The output file is `<name>.png`. |

**Returns** — `None`. Saves `<name>.png` to the current directory.

**Notes**

- Requires `kaleido` to be installed (`pip install kaleido==0.2.1`).
- Same electrode-name requirements as `visualize_html`.

**Example**

```python
G.visualize_png(graphs[0], 'window_0_pearson')
# Saves: window_0_pearson.png
```
