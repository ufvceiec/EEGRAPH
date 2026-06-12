# Visualization

EEGraph generates interactive HTML graphs where electrodes are placed at their anatomical positions on a 2D head map.

---

## Interactive HTML

```python
G.visualize_html(graphs[0], 'my_graph')
# Saves my_graph_plot.html and opens it in the default browser
```

The HTML output provides two display modes, toggled by buttons in the top-left corner:

- **Show edge markers** — small markers appear at the midpoint of each edge; hovering over them displays the connectivity value.
- **Hide edge markers** — cleaner view; values are still accessible on hover.

Edge **width** is proportional to the connectivity value (weight³ × 6), making strong connections visually prominent.

---

## Static PNG

```python
G.visualize_png(graphs[0], 'my_graph')
# Saves my_graph.png (1800 × 1000 px)
```

Requires `kaleido` to be installed (`pip install kaleido`).

---

## Multiple windows

Visualise each time window separately:

```python
for i, graph in graphs.items():
    G.visualize_html(graph, f'graph_window_{i}')
```

---

## Channel name requirements

Electrodes are placed by matching channel labels against a dictionary of **333 positions** from the 10/20, 10/10, and 10/5 systems.

| Format accepted | Example |
|-----------------|---------|
| Standard label | `Fp1`, `Cz`, `Pz` |
| Space-separated | `EEG Fp1` (electrode name on the right) |
| Dash-separated | `Fp1-EEG` (electrode name on the left) |

!!! warning
    Channels not found in the position dictionary are **excluded from the visualisation** and a warning is printed. They are still included in the connectivity matrix.

Supported aliases: `T3 = T7`, `T4 = T8`, `T5 = P7`, `T6 = P6`.

---

## Directed graphs (DTF)

When using the `dtf` connectivity measure, EEGraph builds a **directed graph**. The visualisation adds arrow annotations indicating the direction of information flow between electrodes.

```python
graphs, matrix = G.modelate(
    window_size=5, connectivity='dtf', bands=['alpha']
)
G.visualize_html(graphs[0], 'dtf_graph')
```
