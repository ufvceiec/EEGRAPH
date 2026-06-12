# Loading EEG Data

## Supported formats

| Format | Extension | MNE reader used |
|--------|-----------|-----------------|
| European Data Format | `.edf` | `mne.io.read_raw_edf` |
| General Data Format | `.gdf` | `mne.io.read_raw_gdf` |
| Biosemi Data Format | `.bdf` | `mne.io.read_raw_bdf` |
| Brainvision | `.vhdr` | `mne.io.read_raw_brainvision` |
| Neuroscan CNT | `.cnt` | `mne.io.read_raw_cnt` |
| EGI simple binary | `.egi` | `mne.io.read_raw_egi` |
| EGI MFF format | `.mff` | `mne.io.read_raw_egi` |
| eXimia | `.nxe` | `mne.io.read_raw_eximia` |

---

## Basic loading

```python
import eegraph

G = eegraph.Graph()
G.load_data(path='my_recording.edf')
```

EEGraph automatically:

- Detects the file format from the extension
- Identifies the number of electrodes
- Reads channel labels and places each node in its correct EEG headset position
- Prints recording duration, sample rate, and channel names

---

## Excluding channels

Use the `exclude` parameter to remove channels from the analysis (e.g. ECG, EMG, or reference channels):

```python
G.load_data(
    path='my_recording.edf',
    exclude=['EEG EKG1-EKG2', 'EEG EMG-REF']
)
```

---

## Electrode montage file

Some EEG systems export files without standard channel labels. EEGraph accepts a separate montage file to map electrode numbers to standard labels.

```python
G.load_data(
    path='my_recording.gdf',
    electrode_montage_path='electrodemontage.set.ced'
)
```

The montage file must contain at least two columns:

- One column with the **electrode number** as it appears in the EEG file
- One column with the **standard electrode label** (e.g. `Fp1`, `Cz`)

Accepted delimiters: whitespace, `;`, or `:`.

### Example montage file

```
1  Fp1
2  Fp2
3  F7
4  F3
5  Fz
...
```

---

## Channel name formats

EEGraph recognises three channel name formats for graph visualisation:

| Format | Example |
|--------|---------|
| Standard | `Fp1`, `Cz`, `O2` |
| Space-separated | `EEG Fp1`, `EEG Cz` (electrode name on the right) |
| Dash-separated | `Fp1-EEG`, `Cz-REF` (electrode name on the left) |

!!! note
    Channels not matching any of the 333 known electrode positions will be included in the connectivity analysis but **excluded from visualisation**, with a warning.

---

## Electrode position system

EEGraph contains a dictionary of **333 electrode positions** covering the 10/20, 10/10, and 10/5 systems. Equivalent label aliases are included:

| Alias | Standard |
|-------|----------|
| T3 | T7 |
| T4 | T8 |
| T5 | P7 |
| T6 | P6 |
