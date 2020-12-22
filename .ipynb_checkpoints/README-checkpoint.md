# EEGraph

EEGraph is a Python library to model electroencephalograms (EEGs) as graphs, so the connectivity between different brain areas could be analyzed. It has applications
in the study of neurologic diseases like Parkinson or epilepsy. The graph can be exported as a NetworkX graph-like object or it can also be graphically visualized. 


## Getting Started


### Prerequisites

What libraries you need to install.

* Numpy
* Pandas
* Mne
* Matplotlib
* NetworkX
* Plotly
* Scipy
* Scot
* Entropy
```python
git clone https://github.com/raphaelvallat/entropy.git entropy/
cd entropy/
pip install -r requirements.txt
python setup.py develop
```

### Installing EEGraph

To install the latest stable version of EEGraph, you can use pip in a terminal:

```python
pip install eegraph
```

## Functions

### Documentation
[EEGraph documentation](https://github.com/ufvceiec/EEGRAPH/wiki) is available online.

### Importing EEG data 
The different supported EEG file formats by EEGraph.

##### File format | Extension
* Brainvision | .vhdr
* Neuroscan CNT  | .cnt
* European data format | .edf
* Biosemi data format | .bdf
* General data format | .gdf
* EGI simple binary | .egi
* EGI MFF format | .mff
* eXimia | .nxe

### Connectivity Measures
The different available connectivity measures in EEGraph. 

* Cross Correlation
* Pearson Correlation
* Squared Coherence
* Imaginary Coherence
* Corrected Cross Correlation
* Weighted Phase Lag Index (WPLI)
* Phase Locking Value (PLV)
* Phase Lag Index (PLI)
* Directed Transfer Function (DTF)
* Power Spectrum
* Spectral Entropy
* Shannon Entropy


## Usage
Example usage of the library with Pearson Correlation. 
```python
import eegraph
eegraph.connectivity.pearson_correlation(path="espasmo1.edf", window_size = 2, exclude = ['EEG TAntI1-TAntI', 'EEG TAntD1-TAntD'])
```


### Graph Visualization Example

![Connectivity Graph Output Example](https://github.com/ufvceiec/EEGRAPH/blob/develop/demo/eegraph_output.gif)


## Versioning
See [CHANGELOG.txt](CHANGELOG.txt) for major/breaking updates and version history.


## Authors
Centro de Estudios e Innovación en Gestión del Conocimiento (CEIEC), Universidad Francisco de Vitoria. 

## Contact


## License

This project is licensed under the [MIT](https://choosealicense.com/licenses/mit/) License.

## Acknowledgments






