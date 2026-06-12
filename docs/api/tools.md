# API Reference — `tools`

The `eegraph.tools` module contains pure utility functions. Most users interact with these indirectly through the `Graph` class, but two functions are useful for computing graph-theoretic metrics directly on the NetworkX graphs returned by `modelate()`.

---

## `compute_graph_metrics(G)`

Compute a set of graph-theoretic metrics for a single NetworkX graph.

```python
from eegraph.tools import compute_graph_metrics
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `G` | `networkx.Graph` | A connectivity graph (one entry from the `graphs` dict). |

**Returns** — `dict` with the following keys:

| Key | Description |
|-----|-------------|
| `'nodes'` | Number of nodes (channels) |
| `'edges'` | Number of edges |
| `'density'` | Graph density (edges / possible edges) |
| `'average_clustering'` | Mean clustering coefficient across all nodes |
| `'average_shortest_path'` | Average shortest path length (returns `None` if graph is disconnected) |
| `'betweenness_centrality'` | Dict of node → betweenness centrality |
| `'closeness_centrality'` | Dict of node → closeness centrality |
| `'degree_centrality'` | Dict of node → degree centrality |

**Example**

```python
import eegraph
from eegraph.tools import compute_graph_metrics

G_obj = eegraph.Graph()
G_obj.load_data(path='eeg_sample.edf')
graphs, matrix = G_obj.modelate(window_size=2, connectivity='pearson_correlation')

metrics = compute_graph_metrics(graphs[0])
print(f"Density: {metrics['density']:.4f}")
print(f"Average clustering: {metrics['average_clustering']:.4f}")
```

---

## `compute_metrics_all(graphs)`

Compute `compute_graph_metrics` for every graph in the `graphs` dict returned by `modelate()`.

```python
from eegraph.tools import compute_metrics_all
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `graphs` | `dict` | The full `graphs` dict returned by `Graph.modelate()`. |

**Returns** — `dict` mapping each graph key (same keys as `graphs`) to the metrics dict produced by `compute_graph_metrics`.

**Example**

```python
from eegraph.tools import compute_metrics_all

graphs, matrix = G_obj.modelate(window_size=2, connectivity='pearson_correlation')
all_metrics = compute_metrics_all(graphs)

for key, m in all_metrics.items():
    print(f"Graph {key}: edges={m['edges']}, density={m['density']:.4f}")
```

**Notes**

- This is equivalent to calling `Graph.compute_metrics(graphs)` on the `Graph` object; both ultimately delegate to `compute_graph_metrics`.
- For disconnected graphs (density close to 0), `'average_shortest_path'` will be `None` because NetworkX raises an exception for disconnected graphs; `compute_graph_metrics` catches this and returns `None` for that field.
