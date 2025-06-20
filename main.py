import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
from mne_connectivity import spectral_connectivity_epochs

# ========================================
# 1) 사용자 설정: 파일 경로, 스케일 팩터
# ========================================
file_path = "C:/Users/user/Documents/Workspace/psychiatry/EEG_signal_data_250529/User/YGM 2025-02-10 10.57.04 [최연섭_V0_EC]/raw/2025-02-10 11.00.37.txt"
scaling_factor = 0.001  # μV/bit 예시값

# ========================================
# 2) EEG 데이터 불러오기
# ========================================
# 헤더에서 채널명 탐색
with open(file_path, "r") as f:
    header_lines = [next(f) for _ in range(50)]
for i, line in enumerate(header_lines):
    if "CH0:FRAME" in line and "EEG" in line:
        cols = line.strip().split()
        header_idx = i
        break

exclude = {"CH0:FRAME:N", "CH40:ECG:ECG1", "CH41:TRG:TRG"}
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
# 3) RawArray 생성 및 초기 크롭
# ========================================
data = (df.values.T * scaling_factor)
sfreq = 250
info = mne.create_info(ch_names=chan_names, sfreq=sfreq, ch_types="eeg")
raw = mne.io.RawArray(data, info)

# 표준 montage & 평균참조
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)
raw.set_eeg_reference("average", projection=False)

# **초반 0.5초 제거**
raw.crop(tmin=0.5, tmax=None)

# ========================================
# 4) 필터링: HPF 0.5 Hz, LPF 40 Hz, Notch 60 Hz
# ========================================
raw.filter(l_freq=0.5, h_freq=40.0, fir_design="firwin")
raw.notch_filter(freqs=[60, 120], method="spectrum_fit")

# ========================================
# 5) ICA 아티팩트 제거
# ========================================
ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(raw)
eog_inds, _ = ica.find_bads_eog(raw, ch_name="Fp1")
ica.exclude = eog_inds
raw_clean = raw.copy()
ica.apply(raw_clean)

# ========================================
# 6) Epoch 생성 (0–4s segments)
# ========================================
events = mne.make_fixed_length_events(raw_clean, id=1, duration=4.0)
epochs_clean = mne.Epochs(raw_clean, events, tmin=0.0, tmax=4.0,
                          baseline=None, preload=True)

# ========================================
# 7) PLI 계산 (Theta 4–8 Hz) via mne_connectivity
# ========================================
# 1) Prepare inputs
data = epochs_clean.get_data().astype(np.float64)       # shape: (n_epochs, n_channels, n_times)
names = epochs_clean.ch_names        # list of channel names

# 2) Compute PLI in theta band (4–8 Hz)
conn = spectral_connectivity_epochs(
    data=data,
    names=names,
    method='pli',
    indices=None,
    sfreq=sfreq,
    mode='fourier',
    fmin=4.0,
    fmax=8.0,
    fskip=0,
    faverage=True,
    block_size=1000,
    # tmin=np.array([0.0]),   
    # tmax=np.array([4.0]),   
    n_jobs=1,
    verbose=False
)

# 3) PLI 데이터 불러와서 2D로 변환
pli_matrix = conn.get_data(output='dense')      # shape: (1, n_chan, n_chan)
pli2d      = pli_matrix.squeeze()               # shape: (n_chan, n_chan)

# 4) 대칭화
pli_sym = pli2d + pli2d.T
np.fill_diagonal(pli_sym, 0)

# 5) 대칭성 재검사
asym = np.max(np.abs(pli_sym - pli_sym.T))
print(f"Max asymmetry: {asym:.3e}")

# 6) 히스토그램 (upper triangle values만)
# vals = pli_sym[np.triu_indices_from(pli_sym, k=1)]
# plt.figure()
# plt.hist(vals, bins=50)
# plt.xlabel("PLI")
# plt.ylabel("Count")
# plt.title("Distribution of Theta-band PLI")
# plt.show()

# 7) PLI 매트릭스 시각화
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

# 1) 채널 위치 추출 (2D: x, y)
ch_pos3d = montage.get_positions()['ch_pos']  # {'Fp1':(x,y,z), …}
ch_pos2d = {ch: ch_pos3d[ch][:2] for ch in chan_names}

# 2) MST 그래프(G) 생성 (이전 코드에서 이미 만드셨다면 재사용)
weights = 1.0 - pli_sym       # pli_sym: 대칭화된 PLI matrix
mst_sparse = minimum_spanning_tree(weights)
mst = mst_sparse.toarray()  # (n_chan, n_chan), 0/weight
G = nx.from_numpy_array(mst)

# 3) 노드 인덱스 → 채널명 매핑
mapping = {i: chan_names[i] for i in range(len(chan_names))}
G = nx.relabel_nodes(G, mapping)

# 5) 네트워크 지표 예시
degree_dict    = dict(G.degree())
leaf_fraction  = sum(1 for v in degree_dict.values() if v==1) / (len(G)-1)
diameter       = nx.diameter(G)  # 트리이므로 연결 보장
betweenness    = nx.betweenness_centrality(G)
avg_bc         = np.mean(list(betweenness.values()))

print(f"Leaf fraction: {leaf_fraction:.3f}")
print(f"Diameter: {diameter}")
print(f"Mean betweenness centrality: {avg_bc:.3f}")

# 4) 실제 두피 좌표로 그리기
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

