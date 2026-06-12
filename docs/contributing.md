# Contributing to EEGraph

Thank you for your interest in contributing to EEGraph. This guide covers the development environment, how to add new connectivity measures or graph metrics, how to run tests, and how to update the documentation.

---

## Setting up the development environment

**Prerequisites:** Python 3.8+ (tested on 3.8 and 3.13), Git.

```bash
# 1. Fork and clone the repository
git clone https://github.com/ufvceiec/EEGRAPH.git
cd EEGRAPH
git checkout develop-refactor

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install runtime + development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov

# 4. (Optional) Install MkDocs for documentation preview
pip install mkdocs>=1.5 mkdocs-material>=9.0 mkdocstrings pymdown-extensions>=10.0

# 5. Verify the test suite runs
python -m pytest tests/tests.py -v -k "not (test_load_data or test_modelate or test_visualize)"
```

---

## Adding a new connectivity measure (Strategy pattern)

EEGraph uses the **Strategy pattern** (`eegraph/strategy.py`). Each connectivity measure is a class that inherits from one of six workflow base classes:

| Base class | Use when |
|------------|----------|
| `Connectivity_No_Bands` | Pairwise scalar, no frequency bands |
| `Connectivity_With_Bands` | Pairwise scalar, per frequency band |
| `Connectivity_single_channel_No_Bands` | Single-channel scalar, no bands |
| `Connectivity_single_channel_With_Bands` | Single-channel scalar, per band |
| `Cross_correlation_rescaled` | Cross-correlation variants (rescales input) |
| `Dtf_With_Bands` | Directed Transfer Function via SCoT |

### Step-by-step

**1. Implement the estimator class in `eegraph/strategy.py`:**

```python
class My_measure_Estimator(Connectivity_No_Bands):
    def __init__(self):
        self.threshold = 0.5   # default edge-weight cutoff

    def calculate_conn(self, data, steps, num_channels, *args):
        """
        data      — 1-D array of EEG samples for the current window,
                    ordered [ch0, ch1, ..., chN].
        steps     — list of (start, end) sample indices for this window.
        num_channels — number of EEG channels.

        Returns: np.ndarray of shape (num_channels, num_channels).
        """
        import numpy as np
        result = np.zeros((num_channels, num_channels))
        # ... compute pairwise connectivity ...
        return result
```

For band-based measures, inherit from `Connectivity_With_Bands` and implement `calculate_conn(self, data, steps, num_channels, freq_bands, sample_freq, *args)`.

**2. Register the measure in `eegraph/tools.py`:**

```python
connectivity_measures = {
    # ... existing entries ...
    'my_measure': 'My_measure_Estimator',
}
```

**3. Add tests in `tests/tests.py`:**

```python
class TestMyMeasure(unittest.TestCase):
    def test_basic(self):
        # Load a sample EEG and call modelate with 'my_measure'
        ...
```

**4. Document the measure** in `docs/user_guide/connectivity_measures.md` and add a row to the table in `docs/api/graph.md`.

---

## Adding graph metrics

Graph metrics are implemented in `eegraph/tools.py` inside `compute_graph_metrics(G)`.

**1. Edit `compute_graph_metrics` to add your metric:**

```python
def compute_graph_metrics(G):
    import networkx as nx
    metrics = {}
    metrics['nodes'] = G.number_of_nodes()
    metrics['edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)
    metrics['average_clustering'] = nx.average_clustering(G)
    # Add your metric here:
    metrics['my_metric'] = nx.my_metric(G)
    # ...
    return metrics
```

**2.** The `compute_metrics_all(graphs)` wrapper and `Graph.compute_metrics()` will automatically expose the new metric — no further changes needed.

**3. Add a test** verifying the metric returns a sensible value for a small synthetic graph.

**4. Document** the new key in `docs/api/tools.md`.

---

## Running tests

```bash
# Fast unit tests (no EEG files required)
python -m pytest tests/tests.py -v \
    -k "not (test_load_data or test_modelate or test_visualize)"

# Full suite (requires decrypted test files in tests/)
python -m pytest tests/tests.py -v

# With coverage report
python -m pytest tests/tests.py --cov=eegraph --cov-report=term-missing \
    -k "not (test_load_data or test_modelate or test_visualize)"
```

**Known failures (do not fix without an upstream fix):**

- `TestImportData`, `TestModelData`, `TestVisualizeData` — require encrypted test EEG files.
- `test_calculate_dtf` — broken because `scot 0.2.1` uses `scipy.shape`, removed in SciPy ≥ 1.10.

---

## Documentation

EEGraph uses [MkDocs](https://www.mkdocs.org/) with the [Material](https://squidfunk.github.io/mkdocs-material/) theme.

```bash
# Install documentation dependencies
pip install mkdocs>=1.5 mkdocs-material>=9.0 mkdocstrings pymdown-extensions>=10.0

# Live preview (auto-reloads on file changes)
mkdocs serve

# Build static site into site/
mkdocs build
```

### Documentation structure

```
docs/
  index.md                        — Home page
  getting_started/
    installation.md
    quickstart.md
  user_guide/
    loading_data.md
    connectivity_measures.md
    graph_metrics.md
    visualization.md
    windowing.md                  — Window size & epoch keys
  api/
    graph.md                      — Graph class API reference
    tools.md                      — tools module API reference
  changelog.md
  contributing.md                 — This file
```

When adding a new page, also register it in the `nav:` section of `mkdocs.yml`.

---

## Pull request guidelines

1. Branch off `develop-refactor` using a descriptive name: `feature/my-measure` or `fix/off-by-one`.
2. Keep commits focused; one logical change per commit.
3. Ensure all fast unit tests pass before opening a PR.
4. Update `CHANGELOG.txt` with a brief description under the upcoming version.
5. Update relevant documentation pages and `CLAUDE.md` if architecture changes.
