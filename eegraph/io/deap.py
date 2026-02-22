import pickle
import numpy as np
import mne

DEAP_EEG_CHANNELS = [
    "Fp1","AF3","F3","F7","FC5","FC1","C3","T7","CP5","CP1","P3","P7","PO3","O1",
    "Oz","Pz","Fp2","AF4","Fz","F4","F8","FC6","FC2","Cz","C4","T8","CP6","CP2",
    "P4","P8","PO4","O2"
]

def load_deap_dat(path, eeg_only=True, as_epochs=True, sfreq=128.0):
    """
    Load a DEAP .dat file and return EEG data as an MNE object.

    Parameters
    ----------
    path : str
        Path to the .dat file.
    eeg_only : bool, optional
        If True, keep only the 32 EEG channels (discard peripheral signals).
    as_epochs : bool, optional
        If True, return data as MNE EpochsArray (one epoch per trial).
        If False, concatenate all trials and return as a RawArray.
    sfreq : float, optional
        Sampling frequency of the signals. Default is 128 Hz for DEAP.

    Returns
    -------
    epochs_or_raw : mne.EpochsArray or mne.io.RawArray
        The EEG signals wrapped in an MNE object.
    labels : np.ndarray
        The labels associated with each trial (e.g., arousal, valence).
    """

    with open(path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")  # DEAP .dat requires latin1 encoding
    data = obj["data"]   # shape: (trials, channels, time)
    labels = obj.get("labels")

    # Keep only the 32 EEG channels
    if eeg_only and data.shape[1] >= 32:
        data = data[:, :32, :]

    n_trials, n_ch, n_t = data.shape
    ch_names = DEAP_EEG_CHANNELS[:n_ch]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    if as_epochs:
        # One epoch per trial
        epochs = mne.EpochsArray(data.astype(np.float64), info)
        return epochs, labels
    else:
        # Concatenate all trials into a single Raw object
        raw = mne.io.RawArray(data.reshape(n_trials*n_ch, n_t), info)
        return raw, labels
