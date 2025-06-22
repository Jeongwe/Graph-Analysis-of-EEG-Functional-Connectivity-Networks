import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
from mne_connectivity import spectral_connectivity_epochs

# ========================================
# Setting the path and scaling factor
# ========================================
file_path = "eeg.txt"
scaling_factor = 0.001  # ADC to μV/bit (Depending on your equipment settings)

# ========================================
# Load EEG data
# ========================================
# Search channel name in header
with open(file_path, "r") as f:
    header_lines = [next(f) for _ in range(50)]
for i, line in enumerate(header_lines):
    if "CH0:FRAME" in line and "EEG" in line:
        cols = line.strip().split()
        header_idx = i
        break

exclude = {"CH0:FRAME:N", "CH40:ECG:ECG1", "CH41:TRG:TRG"}    # Delete channels unrelated to EEG
eeg_cols = [c for c in cols if c not in exclude]

df = pd.read_csv(
    file_path,
    delim_whitespace=True,
    skiprows=header_idx + 1,
    names=cols,
    usecols=eeg_cols
)

chan_names = [c.split(":")[-1] for c in eeg_cols]
df.columns = chan_names

# ========================================
# Creating a RawArray and Initial 0.5 sec Crop
# ========================================
data = (df.values.T * scaling_factor)
sfreq = 250
info = mne.create_info(ch_names=chan_names, sfreq=sfreq, ch_types="eeg")
raw = mne.io.RawArray(data, info)

# standard montage & average reference
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)
raw.set_eeg_reference("average", projection=False)

raw.crop(tmin=0.5, tmax=None)

# ========================================
# Filtering: HPF 0.5 Hz, LPF 40 Hz, Notch 60 Hz
# ========================================
raw.filter(l_freq=0.5, h_freq=40.0, fir_design="firwin")
raw.notch_filter(freqs=[60, 120], method="spectrum_fit")

# ========================================
# Delete ICA artifact
# ========================================
ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(raw)
eog_inds, _ = ica.find_bads_eog(raw, ch_name="Fp1")
ica.exclude = eog_inds
raw_clean = raw.copy()
ica.apply(raw_clean)

# ========================================
# Create Epoch 4 sec
# ========================================
events = mne.make_fixed_length_events(raw_clean, id=1, duration=4.0)
epochs_clean = mne.Epochs(raw_clean, events, tmin=0.0, tmax=4.0,
                          baseline=None, preload=True)

# ========================================
# Calculate PLI 
# ========================================
data = epochs_clean.get_data().astype(np.float64)       # shape: (n_epochs, n_channels, n_times)
names = epochs_clean.ch_names       

# Compute PLI in theta band (4–8 Hz)
conn = spectral_connectivity_epochs(
    data=data,
    names=names,
    method='pli',
    indices=None,
    sfreq=sfreq,
    mode='fourier',
    fmin=4.0,        # You can find it for any frequency band (e.g. Alpha, Beta...) by replacing fmin, fmax to match the frequency band.
    fmax=8.0,
    fskip=0,
    faverage=True,
    block_size=1000,
    n_jobs=1,
    verbose=False
)

# Visualization PLI matrix
pli_matrix = conn.get_data(output='dense')      # shape: (1, n_chan, n_chan)
pli2d      = pli_matrix.squeeze()               # shape: (n_chan, n_chan)

pli_sym = pli2d + pli2d.T
np.fill_diagonal(pli_sym, 0)

asym = np.max(np.abs(pli_sym - pli_sym.T))
print(f"Max asymmetry: {asym:.3e}")

plt.figure(figsize=(6,5))
plt.imshow(pli_sym, origin='lower', cmap='viridis', vmin=0, vmax=1)
plt.title("Theta-band PLI (4–8 Hz)")
plt.xlabel("Channel Index")
plt.ylabel("Channel Index")
plt.colorbar(label="PLI")
plt.tight_layout()
plt.show()

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
import matplotlib.pyplot as plt

import networkx as nx

# ========================================
# Create MST graph
# ========================================
# Extract channel position (2D: x, y)
ch_pos3d = montage.get_positions()['ch_pos']  # {'Fp1':(x,y,z), …}
ch_pos2d = {ch: ch_pos3d[ch][:2] for ch in chan_names}

weights = 1.0 - pli_sym       
mst_sparse = minimum_spanning_tree(weights)
mst = mst_sparse.toarray()  # (n_chan, n_chan), 0/weight
G = nx.from_numpy_array(mst)

# Map node index → channel name
mapping = {i: chan_names[i] for i in range(len(chan_names))}
G = nx.relabel_nodes(G, mapping)

# Network metrics
degree_dict    = dict(G.degree())
leaf_fraction  = sum(1 for v in degree_dict.values() if v==1) / (len(G)-1)
diameter       = nx.diameter(G) 
betweenness    = nx.betweenness_centrality(G)
avg_bc         = np.mean(list(betweenness.values()))

print(f"Leaf fraction: {leaf_fraction:.3f}")
print(f"Diameter: {diameter}")
print(f"Mean betweenness centrality: {avg_bc:.3f}")

# eeg topographic map
plt.figure(figsize=(6,6))
nx.draw(
    G,
    pos=ch_pos2d,
    node_size=50,
    edge_color='gray',
    with_labels=True,
    font_size=20
)
plt.title("MST plotted on 2D scalp locations")
plt.axis('off')
plt.show()

