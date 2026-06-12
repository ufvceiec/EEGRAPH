# Connectivity Measures

EEGraph implements **13 connectivity measures** covering both time-domain and frequency-domain approaches. Pass the measure name as the `connectivity` parameter of `modelate()`.

---

## Time-domain measures

These measures do not require frequency bands (`bands` parameter must not be specified).

### Pearson Correlation

**Key:** `pearson_correlation` | **Default threshold:** 0.7 | **Range:** [-1, 1]

Measures the linear correlation between two EEG channels.

$$C_{x,y} = \frac{\sigma_{xy}}{\sigma_x \sigma_y}$$

```python
graphs, matrix = G.modelate(window_size=5, connectivity='pearson_correlation')
```

**Interpretation:** Values near 1 indicate strong co-activation. Symmetric (undirected). Simple and fast; assumes linear relationships.

---

### Cross Correlation

**Key:** `cross_correlation` | **Default threshold:** 0.5 | **Range:** ~[-1, 1]

Measures the similarity between two channels as a function of time lag. The mean of the 10% of positive-lag values is reported.

```python
graphs, matrix = G.modelate(window_size=5, connectivity='cross_correlation')
```

**Interpretation:** Captures delayed linear coupling. Useful when one region drives another with a short time delay.

---

### Corrected Cross Correlation

**Key:** `corr_cross_correlation` | **Default threshold:** 0.1 | **Range:** ~[-1, 1]

Measures the *asymmetry* of the cross-correlation: $CCC_{xy}(m) = CC_{xy}(m) - CC_{xy}(-m)$.  
Input data is mean-subtracted before computation. The mean of the 10% of lag values is reported.

```python
graphs, matrix = G.modelate(window_size=5, connectivity='corr_cross_correlation')
```

**Interpretation:** Quantifies the directionality of the linear coupling — positive values suggest that x leads y.

---

### Phase Lag Index (PLI)

**Key:** `pli` | **Default threshold:** 0.1 | **Range:** [0, 1]

Measures the consistency of the phase difference sign between two channels, ignoring zero-lag interactions.

$$PLI = |\mathbb{E}[\text{sign}(\Delta\phi(t))]|$$

```python
graphs, matrix = G.modelate(window_size=5, connectivity='pli')
```

**Interpretation:** Robust to volume conduction (which produces zero-lag coupling). Values near 1 indicate consistent phase leading/lagging.

---

### Shannon Entropy

**Key:** `shannon_entropy` | **Default threshold:** 0.25 (25%) | **Single-channel measure**

Measures the average information content of a single EEG channel.

$$H(x) = -\sum_i p(x_i) \log(p(x_i))$$

```python
graphs, matrix = G.modelate(window_size=5, connectivity='shannon_entropy')
```

**Interpretation:** Higher entropy indicates more complex, unpredictable signal. The top 25% of channels by entropy value are connected in the graph.

---

## Frequency-domain measures

These measures require the `bands` parameter. Choose one or more from: `'delta'`, `'theta'`, `'alpha'`, `'beta'`, `'gamma'`.

| Band | Range |
|------|-------|
| Delta | 1–4 Hz |
| Theta | 4–8 Hz |
| Alpha | 8–13 Hz |
| Beta | 13–30 Hz |
| Gamma | 30–45 Hz |

---

### Squared Coherence

**Key:** `squared_coherence` | **Default threshold:** 0.65 | **Range:** [0, 1]

Measures the spectral correlation between two channels in each frequency band.

$$SC_{xy}(f) = \frac{|G_{xy}(f)|^2}{G_{xx}(f) G_{yy}(f)}$$

```python
graphs, matrix = G.modelate(
    window_size=5, connectivity='squared_coherence',
    bands=['alpha', 'beta']
)
```

**Interpretation:** Values near 1 indicate synchronised oscillations. Sensitive to volume conduction (zero-lag coupling).

---

### Imaginary Coherence

**Key:** `imag_coherence` | **Default threshold:** 0.4 | **Range:** [-1, 1]

Uses only the imaginary part of the cross-spectrum, making it insensitive to zero-lag (volume-conducted) coupling.

$$IC_{xy}(f) = \frac{\text{Im}(G_{xy}(f))}{\sqrt{G_{xx}(f) G_{yy}(f)}}$$

```python
graphs, matrix = G.modelate(
    window_size=5, connectivity='imag_coherence',
    bands=['alpha', 'beta']
)
```

**Interpretation:** Preferred over squared coherence when volume conduction is a concern.

---

### Phase Locking Value (PLV)

**Key:** `plv` | **Default threshold:** 0.8 | **Range:** [0, 1]

Measures the consistency of the instantaneous phase difference between two channels.

$$PLV = |\mathbb{E}[\exp(i \Delta\phi_{rel}(t))]|$$

```python
graphs, matrix = G.modelate(
    window_size=5, connectivity='plv',
    bands=['theta', 'alpha']
)
```

**Interpretation:** Values near 1 indicate phase synchrony. Sensitive to volume conduction; prefer PLI or WPLI for source-level analysis.

---

### Phase Lag Index with bands (PLI bands)

**Key:** `pli_bands` | **Default threshold:** 0.1 | **Range:** [0, 1]

Same as PLI but applied per frequency band via FFT decomposition.

```python
graphs, matrix = G.modelate(
    window_size=5, connectivity='pli_bands',
    bands=['alpha', 'beta']
)
```

---

### Weighted Phase Lag Index (WPLI)

**Key:** `wpli` | **Default threshold:** 0.45 | **Range:** [0, 1]

Weights phase-lag contributions by the magnitude of the imaginary cross-spectrum, reducing bias from noise.

$$WPLI = \frac{|\mathbb{E}[|\text{Im}(G_{xy})| \cdot \text{sign}(\text{Im}(G_{xy}))]|}{\mathbb{E}[|\text{Im}(G_{xy})|]}$$

```python
graphs, matrix = G.modelate(
    window_size=5, connectivity='wpli',
    bands=['theta', 'alpha', 'beta']
)
```

**Interpretation:** More robust than PLI to noise and small sample sizes. Preferred for resting-state connectivity studies.

---

### Directed Transfer Function (DTF)

**Key:** `dtf` | **Default threshold:** 0.3 | **Range:** [0, 1] | **Directed graph**

Describes the causal influence of channel j on channel i using a multivariate autoregressive (MVAR) model.

$$DTF^2_{j \to i}(f) = \frac{|H_{ij}(f)|^2}{\sum_m |H_{im}(f)|^2}$$

```python
graphs, matrix = G.modelate(
    window_size=5, connectivity='dtf',
    bands=['alpha', 'beta']
)
```

!!! warning
    DTF requires `scot 0.2.1`, which is incompatible with SciPy ≥ 1.10. This measure is currently unavailable on newer SciPy versions.

**Interpretation:** Values near 1 indicate strong directed information flow. Produces a **directed graph** (DiGraph in NetworkX).

---

### Power Spectrum

**Key:** `power_spectrum` | **Default threshold:** 0.25 (25%) | **Single-channel measure**

Computes the mean power in each frequency band per channel using the FFT.

$$PS(f) = |X(f)|^2$$

```python
graphs, matrix = G.modelate(
    window_size=5, connectivity='power_spectrum',
    bands=['delta', 'theta', 'alpha']
)
```

**Interpretation:** The top 25% of channels by band power are connected in the graph. Useful for identifying regions with dominant oscillatory activity.

---

### Spectral Entropy

**Key:** `spectral_entropy` | **Default threshold:** 0.25 (25%) | **Single-channel measure**

Measures the complexity of the power spectrum in each frequency band (via Welch's method), normalised to [0, 1].

```python
graphs, matrix = G.modelate(
    window_size=5, connectivity='spectral_entropy',
    bands=['alpha', 'beta']
)
```

**Interpretation:** Low spectral entropy indicates a dominant frequency (regular signal). High entropy indicates broadband activity (complex signal). The top 25% of channels are connected.

---

## Custom threshold

All measures accept an optional `threshold` parameter that overrides the default:

```python
graphs, matrix = G.modelate(
    window_size=5,
    connectivity='pearson_correlation',
    threshold=0.85
)
```

!!! tip
    For measures with a percentage threshold (`power_spectrum`, `spectral_entropy`, `shannon_entropy`), the threshold represents the fraction of top nodes to connect, e.g. `threshold=0.25` connects the top 25%.
