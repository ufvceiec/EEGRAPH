# Window Size & Epochs

EEGraph splits a continuous EEG recording into non-overlapping windows (epochs) before computing connectivity. Understanding how windows map to the output `graphs` dict is essential for downstream analysis.

---

## The `window_size` parameter

`window_size` is passed to `Graph.modelate()` and controls how the recording is divided.

### Integer — uniform windows (seconds)

```python
graphs, matrix = G.modelate(window_size=2, connectivity='pearson_correlation')
```

The recording is divided into consecutive, non-overlapping segments of `window_size` seconds. If the recording length is not an exact multiple of `window_size`, the last partial segment is **dropped**.

**Example:** a 10-second EEG with `window_size=3` produces three windows covering [0–3 s], [3–6 s], and [6–9 s]. The final 1-second tail is discarded.

### List — explicit interval boundaries (seconds)

```python
graphs, matrix = G.modelate(window_size=[0, 3, 8], connectivity='pearson_correlation')
```

The list defines boundary points. `N` values produce `N-1` windows. The first element **must** be `0`.

| List | Windows created |
|------|----------------|
| `[0, 3, 8]` | [0–3 s], [3–8 s] |
| `[0, 5]` | [0–5 s] |
| `[0, 2, 4, 6]` | [0–2 s], [2–4 s], [4–6 s] |

---

## Keys of the returned `graphs` dict

`Graph.modelate()` always returns a `dict`. The key scheme depends on whether frequency bands are used.

### Time-domain (no bands)

Keys are plain integers starting at `0`:

```
graphs = {0: <Graph>, 1: <Graph>, 2: <Graph>, ...}
```

Window `k` covers the k-th time segment.

### Frequency-domain (with bands)

Keys are strings combining window index and band name, separated by `_`:

```
graphs = {
    '0_delta': <Graph>,
    '0_theta': <Graph>,
    '0_alpha': <Graph>,
    '1_delta': <Graph>,
    '1_theta': <Graph>,
    '1_alpha': <Graph>,
    ...
}
```

The format is `"<window_index>_<band_name>"`. Band names match those passed to `bands=`.

---

## Connectivity matrix shape

`modelate()` also returns `connectivity_matrix`, a NumPy array:

```
shape: (num_graphs, num_channels, num_channels)
```

where `num_graphs = num_windows` for time-domain measures, and `num_graphs = num_windows × num_bands` for frequency-domain measures. `connectivity_matrix[i]` is the full (pre-threshold) pairwise connectivity matrix for the i-th graph in iteration order.

---

## Iterating over windows — example code

### Time-domain iteration

```python
import eegraph
import numpy as np

G = eegraph.Graph()
G.load_data(path='eeg_sample.edf')

graphs, matrix = G.modelate(window_size=2, connectivity='pearson_correlation')

for window_idx, graph in graphs.items():
    print(f"Window {window_idx}: {graph.number_of_nodes()} nodes, "
          f"{graph.number_of_edges()} edges")
    # Access the raw connectivity matrix for this window
    conn = matrix[window_idx]   # shape: (num_channels, num_channels)
    print(f"  Mean connectivity: {np.mean(conn):.4f}")
```

### Frequency-domain iteration

```python
graphs, matrix = G.modelate(
    window_size=2,
    connectivity='squared_coherence',
    bands=['alpha', 'beta']
)

for key, graph in graphs.items():
    window_idx, band = key.split('_', 1)
    print(f"Window {window_idx}, band {band}: {graph.number_of_edges()} edges")
```

### Grouping by window across bands

```python
from collections import defaultdict

by_window = defaultdict(dict)
for key, graph in graphs.items():
    w, band = key.split('_', 1)
    by_window[int(w)][band] = graph

for w, band_graphs in sorted(by_window.items()):
    print(f"Window {w}:")
    for band, g in band_graphs.items():
        print(f"  {band}: {g.number_of_edges()} edges")
```

### Saving connectivity matrices per window

```python
import numpy as np

graphs, matrix = G.modelate(window_size=5, connectivity='pearson_correlation')
for i in range(len(graphs)):
    np.savetxt(f'conn_window_{i}.csv', matrix[i], delimiter=',')
```

---

## Notes

- If the EEG duration is shorter than one window, EEGraph internally pads the signal with a duplicate interval and removes the duplicate from the output. The returned `graphs` dict will contain a single entry.
- Channel order in `connectivity_matrix` matches the channel order returned by MNE after applying any `exclude` list.
