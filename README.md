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

### Installing EEGraph

To install the latest stable version of EEGraph, you can use pip in a terminal:

```python
pip install EEGRAPH
```

## Functions

### Documentation
[EEGraph documentation](https://github.com/ufvceiec/EEGRAPH/wiki) is available online. [Examples](https://github.com/ufvceiec/EEGRAPH/blob/develop-refactor/Examples/EEGraph_Example.ipynb) of usage are also available.

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
The different available connectivity measures in EEGraph. Visit [documentation](https://github.com/ufvceiec/EEGRAPH/wiki/Modelate-Data) for more info.

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
***
### Load data
```python
import eegraph
G = eegraph.Graph()
G.load_data(path = "espasmo1.edf", exclude = ['EEG TAntI1-TAntI', 'EEG TAntD1-TAntD', 'EEG EKG1-EKG2'])
```
#### Electrode Montage
An electrode montage file can be specified for channels names while loading EEG data. Visit [documentation](https://github.com/ufvceiec/EEGRAPH/wiki/Load-data-from-EEG) for more info.
```python
import eegraph
G = eegraph.Graph()
G.load_data(path = "espasmo1.edf", electrode_montage_path = 'electrodemontage.set.ced')
```
***
### Modelate data
##### Without frequency bands
```python
graphs, connectivity_matrix = G.modelate(window_size = 2, connectivity = 'pearson_correlation')
```
##### With frequency bands
```python
graphs, connectivity_matrix = G.modelate(window_size = 2, connectivity = 'squared_coherence', bands = ['delta','theta','alpha'])
```
### Threshold
A custom threshold can be specified as a parameter in modelate. Default threshold values can be found in the [documentation](https://github.com/ufvceiec/EEGRAPH/wiki/Modelate-Data).
```python
graphs, connectivity_matrix = G.modelate(window_size = 2, connectivity = 'pearson_correlation', threshold = 0.8)
```
### Window size
The window size can be defined as an _int_ or _list_. 

_int_: The set window size in seconds, e.g.(2). All the time intervals will be 2 seconds long.

_list_: The specific time intervals in seconds, e.g.[0, 3, 8]. The time intervalls will be the same as specified in the input. 
***
### Visualize graph
In order to visualize graphs, EEG channel names must be in one of the following formats:
* Standard: 'Fp1', 'Fp2', 'C3', 'Cz'...
* Dash separated: 'Fp1-EEG', 'Fp2-EEG', 'C3-EEG', 'Cz-EEG'...
* Space separated: 'EEG Fp1', 'EEG Fp2', 'EEG C3', 'EEG Cz'...

For the Space separtor the information on the left side of the separator will be ignored, the standard electrode name must be on the right side. 

For the Dash separtor the information on the right side of the separator will be ignored, the standard electrode name must be on the left side. 

```python
G.visualize(graphs[0], 'graph_1')
```
![Connectivity Graph Output Example](https://github.com/ufvceiec/EEGRAPH/blob/develop/demo/eegraph_output.gif)

## EEGraph Workflow
![EEGraph Workflow Example](https://github.com/ufvceiec/EEGRAPH/blob/develop-refactor/demo/eegraph_workflow.png)
 
## Contributing
See [Contribution guidelines](https://github.com/ufvceiec/EEGRAPH/blob/develop-refactor/CONTRIBUTING.md) for more information.

## Versioning
See [CHANGELOG.txt](CHANGELOG.txt) for major/breaking updates and version history.

## Contact
Centro de Estudios e Innovación en Gestión del Conocimiento (CEIEC), Universidad Francisco de Vitoria.
* Responsible: Alberto Nogales (alberto.nogales@ceiec.es)
* Supervisor: Ana María Maitín
* Main developer: Pedro Chazarra

## Disclaimer
* External dependency 'Entropy' can´t be installed with pip. The installation will happen when EEGraph is imported, a message will appear in the command line asking for permissions to install the missing dependency. 


* There is a known issue with DTF connectivity, sometimes the error 'math domain error' is shown, it is part of an external library.
## License

This project is licensed under the MIT [License](https://github.com/ufvceiec/EEGRAPH/blob/develop-refactor/LICENSE).






