# EEGraph

<p align="center">
  <img src="assets/eegraph-logo.png" alt="EEGraph Logo" width="300"/>
</p>

**EEGraph** is an open-source Python library that automatically models electroencephalograms (EEGs) as graphs, enabling brain connectivity analysis with a single workflow.

Nodes represent EEG electrodes (up to 333 positions from the 10/20, 10/10 and 10/5 systems). Edges are determined by one of **13 connectivity measures**, including both time-domain and frequency-domain approaches. Once the graph is built, **11 graph-theoretic metrics** can be computed directly to characterise the network topology.

---

## Key features

- **One-line workflow** — load → modelate → visualise
- **13 connectivity measures** covering time and frequency domains
- **11 graph-theoretic metrics** with brain connectivity interpretation
- **8 EEG file formats** supported (EDF, GDF, BDF, VHDR, CNT, EGI, MFF, NXE)
- **Interactive HTML visualisation** of the connectivity graph
- **Flexible windowing** — uniform windows or custom epoch boundaries
- **Frequency band analysis** — delta, theta, alpha, beta, gamma

---

## Citation

If you use EEGraph in your research, please cite:

> Maitin, A. M., Nogales, A., Chazarra, P., & García-Tejedor, Á. J. (2023).
> EEGraph: An open-source Python library for modeling electroencephalograms using graphs.
> *Neurocomputing*, 519, 127–134.
> [https://doi.org/10.1016/j.neucom.2022.11.050](https://doi.org/10.1016/j.neucom.2022.11.050)

---

## Quick example

```python
import eegraph

# 1. Load EEG
G = eegraph.Graph()
G.load_data(path='my_recording.edf')

# 2. Build connectivity graph (5-second windows, Pearson correlation)
graphs, matrix = G.modelate(window_size=5, connectivity='pearson_correlation')

# 3. Compute graph-theoretic metrics
metrics = G.compute_metrics(graphs[0])
print(f"Small-world sigma: {metrics['small_world_sigma']:.3f}")
print(f"Global efficiency: {metrics['global_efficiency']:.3f}")

# 4. Visualise
G.visualize_html(graphs[0], 'my_graph')
```

---

## Workflow

![EEGraph Workflow](https://github.com/ufvceiec/EEGRAPH/blob/develop-refactor/demo/eegraph_workflow.png)

---

## Contact

**CEIEC Research Institute, Universidad Francisco de Vitoria, Madrid, Spain**

- Ana María Maitín — [a.maitin@ceiec.es](mailto:a.maitin@ceiec.es)
- Alberto Nogales — [alberto.nogales@ceiec.es](mailto:alberto.nogales@ceiec.es)
