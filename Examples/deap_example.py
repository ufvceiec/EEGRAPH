# Examples/deap_example.py
from pathlib import Path
from eegraph.io.deap import load_deap_dat

# Optional: import Graph if available in your current branch/version
try:
    from eegraph.graph import Graph
    HAS_GRAPH = True
except Exception:
    HAS_GRAPH = False


def main():
    # Path to a DEAP .dat file (use a real subject .dat or the fake one created for testing)
    dat_path = Path("Examples/data/fake_deap.dat")
    if not dat_path.exists():
        raise FileNotFoundError(
            f"{dat_path} not found. Please place a DEAP .dat file here "
            "(e.g., s01.dat) or generate a fake one for testing."
        )

    # 1) Load DEAP .dat -> MNE EpochsArray (5 trials x 32 EEG channels x 1280 samples in the fake example)
    epochs, labels = load_deap_dat(str(dat_path), eeg_only=True, as_epochs=True, sfreq=128.0)
    print(epochs)
    print("labels shape:", None if labels is None else labels.shape)
    print("data shape:", epochs.get_data().shape)
    print("nchan:", epochs.info["nchan"])

    # 2) (Optional) Integrate with EEGRAPH Graph if available
    if HAS_GRAPH:
        try:
            G = Graph()
            # If your Graph API expects MNE objects directly:
            if hasattr(G, "load_data"):
                G.load_data(epochs)  # or G.load_data(epochs, on="epochs") depending on your API
            # Build a connectivity graph (choose one method your lib supports)
            if hasattr(G, "modelate"):
                # Replace 'plv' with another method if needed (e.g., 'coherence', 'pearson', etc.)
                G.modelate(method="plv")
            # Visualize if available in your version
            if hasattr(G, "visualize"):
                G.visualize()
        except Exception as e:
            print("[WARN] Skipping Graph integration due to:", repr(e))
    else:
        print("[INFO] eegraph.graph.Graph not available in this version/branch. Loader test completed.")


if __name__ == "__main__":
    main()
