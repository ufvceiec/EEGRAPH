# EEGraph

EEGraph is a Python library to model electroencephalograms (EEGs) as graphs, so the connectivity between different brain areas could be analyzed. It has applications
in the study of neurologic diseases like Parkinson or epilepsy. The graph can be exported as a NetworkX graph-like object and it can also be visualized. 


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What libraries you need to install and how to install them.

* Numpy
```python
pip install numpy
```
* Pandas
```python
pip install pandas
```
* Mne
```python
pip install -U mne
```
* Matplotlib
```python
pip install matplotlib
```
* NetworkX
```python
pip install networkx
```
* Plotly
```python
pip install plotly
```
* Scipy
```python
pip install scipy
```
* Entropy
```python
git clone https://github.com/raphaelvallat/entropy.git entropy/
cd entropy/
pip install -r requirements.txt
python setup.py develop
```
* Scot
```python
pip install scot
```

### Installing EEGraph

To install the latest stable version of EEGraph, you can use pip in a terminal:

```python
pip install -U eegraph
```

## Functions

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
The different available connectivity measures. 

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
```python
import eegraph
eegraph.connectivity.pearson_correlation(path="espasmo1.edf", window_size = [0,4], exclude = ['EEG TAntI1-TAntI', 'EEG TAntD1-TAntD', 'EEG EKG1-EKG2'])
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With


## Versioning



## Authors



## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

