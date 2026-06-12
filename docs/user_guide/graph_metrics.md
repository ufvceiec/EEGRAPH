# Graph Metrics

After building a connectivity graph with `modelate()`, EEGraph can compute a comprehensive set of graph-theoretic metrics on any resulting NetworkX graph. These metrics quantify the **topology** of the brain network and have well-established interpretations in the neuroscience literature.

---

## Usage

```python
graphs, matrix = G.modelate(window_size=5, connectivity='pearson_correlation')

# All windows at once
all_metrics = G.compute_metrics(graphs)

# Single window
m = G.compute_metrics(graphs[0])

# Access individual metrics
print(m['small_world_sigma'])
print(m['global_efficiency'])
print(m['betweenness_centrality'])   # dict {electrode: value}
```

!!! note
    All metrics are computed on an **undirected, self-loop-free** copy of the graph, regardless of whether the original is directed (DTF) or contains self-loops (single-channel measures).

---

## Network-level metrics

### Density

**Range:** [0, 1]

Fraction of all possible electrode pairs that share an active connection above the chosen threshold.

**Brain connectivity interpretation:**
Overall synchronisation level of the network. Higher density is observed during cognitive tasks or epileptic seizures (hyper-synchrony). Abnormally low density is reported in Alzheimer's disease and disorders of consciousness.

!!! warning
    Density depends directly on the threshold. Always compare recordings using the **same threshold value**.

---

### Transitivity

**Range:** [0, 1]

Global clustering coefficient — ratio of closed triangles to all connected triplets across the whole network.

**Brain connectivity interpretation:**
Captures the tendency of electrodes to form tightly interconnected local clusters (functional modules). The healthy resting brain has high transitivity, reflecting local specialisation. Reduced transitivity has been linked to schizophrenia and TBI; elevated transitivity appears in some forms of epilepsy.

---

### Average Clustering

**Range:** [0, 1]

Mean of each node's local clustering coefficient, weighted by edge strength.

**Brain connectivity interpretation:**
Similar to transitivity but gives equal weight to every electrode regardless of degree. Useful for detecting localised changes in functional modularity. Decreased average clustering in the alpha band has been reported in Alzheimer's disease and Parkinson's disease.

---

### Global Efficiency

**Range:** [0, 1]

Average inverse shortest path length over all node pairs. Well-defined for disconnected graphs (isolated pairs contribute 0).

**Brain connectivity interpretation:**
Reflects how efficiently information is integrated across the whole brain (parallel processing capacity). The healthy brain maintains high global efficiency. Reductions are a consistent finding in Alzheimer's disease, multiple sclerosis, and following stroke. An increase may indicate seizure-related hyper-connectivity.

---

### Local Efficiency

**Range:** [0, 1]

Average global efficiency computed within each node's immediate neighbourhood.

**Brain connectivity interpretation:**
Measures fault tolerance and local information-processing capacity. High local efficiency means the network remains functional even if individual electrodes are compromised. Decreased local efficiency has been observed in Parkinson's disease (particularly in the beta band) and in ageing.

---

### Average Path Length

**Range:** [≥ 1]

Mean number of edges along the shortest path between all pairs of nodes. Computed on the largest connected component when the graph is disconnected; `NaN` if the graph has fewer than 2 connected nodes.

**Brain connectivity interpretation:**
Reflects global integration — how rapidly activity in one brain region can influence another. The healthy brain shows short path lengths (rapid integration). Increased path length indicates fragmented or less integrated connectivity and is reported in Alzheimer's disease, depression, and following white-matter lesions.

---

### Degree Assortativity

**Range:** [-1, 1]

Pearson correlation between the degrees of connected node pairs. Positive: hubs connect to hubs. Negative: hubs connect to peripheral nodes.

**Brain connectivity interpretation:**
Most biological brain networks are **disassortative** (negative values) — hub electrodes connect to peripheral ones, supporting robustness and integration. A shift towards assortativity may reflect pathological reorganisation such as that seen in epilepsy or following focal lesions.

---

### Small-World Sigma (σ)

$$\sigma = \frac{C / C_{rand}}{L / L_{rand}}$$

where $C_{rand} \approx \langle k \rangle / n$ and $L_{rand} \approx \ln(n) / \ln(\langle k \rangle)$ are analytical approximations for a random graph.

**σ > 1** indicates small-world organisation. `NaN` if not computable.

**Brain connectivity interpretation:**
The **most important summary metric** for EEG brain network studies. The healthy brain operates as a small-world network (high clustering + short path lengths), striking an optimal balance between local specialisation and global integration. σ significantly above 1 has been confirmed in resting EEG across all frequency bands.

**Loss of small-world topology** (σ approaching 1) is a robust biomarker reported in:

- Alzheimer's disease
- Schizophrenia
- Epilepsy
- Major depression

---

## Node-level metrics

These return a **dictionary `{electrode_label: value}`**, allowing you to identify which specific electrodes are most central.

### Degree Centrality

**Range:** [0, 1]

Fraction of other electrodes each node is directly connected to.

**Brain connectivity interpretation:**
Identifies electrodes with the largest number of active connections. Frontal and parietal hub electrodes typically show the highest degree centrality in the healthy resting brain. Shifts in degree centrality (e.g. from frontal to temporal regions) can indicate pathological reorganisation in epilepsy or dementia.

```python
most_connected = max(m['degree_centrality'], key=m['degree_centrality'].get)
print(f"Most connected electrode: {most_connected}")
```

---

### Betweenness Centrality

**Range:** [0, 1]

Fraction of all shortest paths that pass through each node, weighted by edge strength.

**Brain connectivity interpretation:**
Identifies **bottleneck electrodes** — those that mediate communication between otherwise distant brain regions. Loss of hub status in key regions (e.g. precuneus, posterior cingulate in Alzheimer's) or the emergence of new hubs (e.g. perilesional areas after stroke) are clinically meaningful signatures.

```python
hub = max(m['betweenness_centrality'], key=m['betweenness_centrality'].get)
print(f"Main hub electrode: {hub}")
```

---

### Eigenvector Centrality

**Range:** [0, 1]

A node's influence weighted by the centrality of its neighbours — being connected to important nodes increases a node's own centrality. Returns `NaN` per node if power iteration fails to converge.

**Brain connectivity interpretation:**
Captures global hub status more sensitively than degree centrality because it accounts for the *quality*, not just the quantity, of connections. Electrodes overlying the default mode network (medial frontal, posterior parietal) show high eigenvector centrality at rest. Reduced eigenvector centrality in these regions has been associated with cognitive decline and Alzheimer's disease.

---

## Group-level analysis recommendations

!!! tip "Threshold normalisation"
    When comparing metrics across subjects or conditions, use **proportional thresholding** (fixing density to the same value across all recordings) to ensure that observed differences reflect genuine topological changes rather than differences in overall connectivity strength.

!!! tip "Multiple windows"
    Use `compute_metrics(graphs)` (passing the full dict) to obtain metrics for every time window. You can then track how network topology evolves over the recording.

    ```python
    import pandas as pd

    all_metrics = G.compute_metrics(graphs)
    df = pd.DataFrame({
        'window': list(all_metrics.keys()),
        'small_world_sigma': [m['small_world_sigma'] for m in all_metrics.values()],
        'global_efficiency': [m['global_efficiency'] for m in all_metrics.values()],
        'average_clustering': [m['average_clustering'] for m in all_metrics.values()],
    })
    print(df)
    ```

---

## References

- Rubinov, M. & Sporns, O. (2010). Complex network measures of brain connectivity: uses and interpretations. *NeuroImage*, 52(3), 1059–1069.
- Stam, C.J. et al. (2007). Small-world networks and functional connectivity in Alzheimer's disease. *Cerebral Cortex*, 17, 92–99.
- Bassett, D.S. & Bullmore, E. (2006). Small-world brain networks. *The Neuroscientist*, 12(6), 512–523.
- Latora, V. & Marchiori, M. (2001). Efficient behavior of small-world networks. *Physical Review Letters*, 87(19), 198701.
