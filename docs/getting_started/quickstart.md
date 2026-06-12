# Quick Start

This page walks through a complete EEGraph workflow in under 10 lines of code.

---

## 1. Load an EEG file

```python
import eegraph

G = eegraph.Graph()
G.load_data(path='my_recording.edf')
```

EEGraph will print the number of channels, sample rate, and duration automatically.

---

## 2. Build a connectivity graph

### Without frequency bands
```python
graphs, matrix = G.modelate(window_size=5, connectivity='pearson_correlation')
```

### With frequency bands
```python
graphs, matrix = G.modelate(
    window_size=5,
    connectivity='squared_coherence',
    bands=['delta', 'theta', 'alpha']
)
```

- `graphs` — dict of NetworkX graph objects, one per window (× bands)
- `matrix` — numpy array of shape `(num_graphs, num_channels, num_channels)`

---

## 3. Compute graph metrics

```python
# All windows
all_metrics = G.compute_metrics(graphs)

# Single window
m = G.compute_metrics(graphs[0])
print(f"Small-world sigma : {m['small_world_sigma']:.3f}")
print(f"Global efficiency : {m['global_efficiency']:.3f}")
print(f"Average clustering: {m['average_clustering']:.3f}")

# Node-level hub detection
top_hub = max(m['betweenness_centrality'], key=m['betweenness_centrality'].get)
print(f"Most central electrode: {top_hub}")
```

---

## 4. Visualise

```python
# Interactive HTML (opens in browser)
G.visualize_html(graphs[0], 'my_graph')

# Static PNG
G.visualize_png(graphs[0], 'my_graph')
```

---

## 5. Save the connectivity matrix

```python
import numpy as np
np.savetxt('connectivity_window0.csv', matrix[0], delimiter=',')
```

---

## Full example

```python
import eegraph
import numpy as np

# Load
G = eegraph.Graph()
G.load_data(path='recording.edf', exclude=['EEG EKG-REF'])

# Modelate
graphs, matrix = G.modelate(
    window_size=5,
    connectivity='pearson_correlation',
    threshold=0.8
)

# Graph metrics for every window
all_metrics = G.compute_metrics(graphs)
for i, m in all_metrics.items():
    print(f"Window {i}: σ={m['small_world_sigma']:.2f}  "
          f"E_glob={m['global_efficiency']:.2f}  "
          f"C={m['average_clustering']:.2f}")

# Visualise window 0
G.visualize_html(graphs[0], 'graph_window_0')

# Save all matrices
np.save('connectivity_matrix.npy', matrix)
```
