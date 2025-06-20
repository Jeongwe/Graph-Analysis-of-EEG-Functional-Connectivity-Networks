import os
import pandas as pd
import numpy as np
import mne
from mne_connectivity import spectral_connectivity_epochs
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
from scipy.stats import ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

# ====================================================
# 1) 피험자 정보 읽기
# ====================================================
subjects_df = pd.read_csv("C:/Users/user/Documents/Workspace/BioNet/matched_patients_eeg_insomnia_EO_amount.csv")   # 불면증: 15명, 정상군: 15명

# Insomnia 컬럼에 NaN 혹은 빈 값이 있는 행은 제외
subjects_df = subjects_df.dropna(subset=["Insomnia"])

# ====================================================
# 2) 한 명분 EEG 파일을 읽어 MST 지표를 리턴하는 함수
# ====================================================
def compute_mst_metrics(eeg_path, scaling_factor=0.001, sfreq=250.0):
    # --- 2.1) 헤더 스캔하여 채널명 찾기 ---
    try:
        with open(eeg_path, "r") as f:
            header = [next(f) for _ in range(50)]
    except (StopIteration, FileNotFoundError) as e:
        print(f"헤더 읽기 실패 ({e}), 파일 건너뜁니다: {eeg_path}")
        return None

    # — 채널 이름 탐색 —
    header_idx = None
    for i, line in enumerate(header):
        if "CH0:FRAME" in line and "EEG" in line:
            cols = line.strip().split()
            header_idx = i
            break
    if header_idx is None:
        print(f"EEG 헤더 라인 미발견, 파일 건너뜁니다: {eeg_path}")
        return None
    
    # 제외할 채널명
    exclude = {"CH0:FRAME:N", "CH40:ECG:ECG1", "CH41:TRG:TRG"}
    eeg_cols = [c for c in cols if c not in exclude]
    # --- 2.2) ASCII → DataFrame ---
    df = pd.read_csv(
        eeg_path,
        delim_whitespace=True,
        skiprows=header_idx + 1,
        names=cols,
        usecols=eeg_cols
    )
    # 채널명 only 마지막 토큰
    chan_names = [c.split(":")[-1] for c in eeg_cols]
    df.columns = chan_names

    # --- 2.3) RawArray 생성 및 프리프로세싱 ---
    data = (df.values.T * scaling_factor)  # 단위: μV
    info = mne.create_info(ch_names=chan_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    mont = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(mont, match_case=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    raw.crop(tmin=0.5, tmax=None)  # 앞 0.5초 제거
    raw.filter(l_freq=0.5, h_freq=40.0, fir_design="firwin", verbose=False)
    raw.notch_filter(freqs=[60, 120], method="spectrum_fit", verbose=False)

    # --- 2.4) ICA로 EOG 아티팩트 제거 ---
    ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800, verbose=False)
    ica.fit(raw)
    eog_inds, _ = ica.find_bads_eog(raw, ch_name="Fp1")
    ica.exclude = eog_inds
    raw_clean = raw.copy()
    ica.apply(raw_clean)

    # --- 2.5) 4s 고정 길이 epoch 생성 ---
    events = mne.make_fixed_length_events(raw_clean, id=1, duration=4.0)
    epochs = mne.Epochs(raw_clean, events, tmin=0.0, tmax=4.0,
                        baseline=None, preload=True, verbose=False)

    # --- 2.6) Theta-band PLI 계산 (4–8Hz) ---
    data_epochs = epochs.get_data().astype(np.float64)  # (n_epochs, n_ch, n_times)
    conn = spectral_connectivity_epochs(
        data_epochs, names=epochs.ch_names,
        method='pli', sfreq=sfreq, mode='fourier',
        # fmin=0.5, fmax=4.0, # Delta
        # fmin=4.0, fmax=8.0, # Theta
        # fmin=8.0, fmax=13.0, # Alpha
        fmin=13.0, fmax=30.0, # Beta
        faverage=True, block_size=1000, n_jobs=1, verbose=False
    )
    pli_mat = conn.get_data(output='dense').squeeze()  # (n_ch, n_ch)
    # 대칭화 및 대각 0
    pli_sym = pli_mat + pli_mat.T
    np.fill_diagonal(pli_sym, 0)

    # --- 2.7) MST 구축 & 지표 계산 ---
    weights = 1.0 - pli_sym
    mst_sparse = minimum_spanning_tree(weights)
    mst = mst_sparse.toarray()
    G = nx.from_numpy_array(mst)

    # 채널 인덱스→이름 매핑
    mapping = {i: epochs.ch_names[i] for i in range(len(epochs.ch_names))}
    G = nx.relabel_nodes(G, mapping)

    # 지표
    deg = dict(G.degree())
    leaf_frac = sum(1 for v in deg.values() if v == 1) / (len(G) - 1)
    diam = nx.diameter(G)
    bc = nx.betweenness_centrality(G)

    metrics = {
        "leaf_fraction":    leaf_frac,
        "diameter":         diam,
        "mean_betweenness": np.mean(list(bc.values()))
    }
    
    # 채널별 BC를 'BC_<채널명>' 키로 추가
    for ch_name, bc_val in bc.items():
        metrics[f"BC_{ch_name}"] = bc_val
    
        # --- MST 시각화 (한 명에 대해서만 plot)---
    # try:
    #     plt.figure(figsize=(6,6))

    #     # 채널 위치: 2D로 투영된 scalp 위치
    #     pos_3d = mont.get_positions()['ch_pos']
    #     pos_2d = {ch: pos_3d[ch][:2] for ch in epochs.ch_names}

    #     nx.draw(
    #         G,
    #         pos=pos_2d,
    #         with_labels=True,
    #         node_size=100,
    #         node_color='skyblue',
    #         edge_color='gray',
    #         font_size=8
    #     )
    #     plt.title(f"MST of EEG (Delta PLI)\n{os.path.basename(eeg_path)}")
    #     plt.axis('off')
    #     plt.tight_layout()
    #     # 저장 (옵션)
    #     save_name = os.path.basename(eeg_path).replace('.txt', '_mst.png')
    #     plt.savefig(os.path.join("MST_figures", save_name), dpi=150)
    #     plt.close()
    # except Exception as e:
    #     print(f"MST 시각화 실패: {e}")

    return metrics

# ====================================================
# 3) 전 피험자에 걸쳐 MST 지표 수집
# ====================================================
metrics_list = []
for _, row in subjects_df.iterrows():
    pid = row["Patient"]
    grp = row["Insomnia"]
    path = row["EEG_path"]
    if not os.path.isfile(path):
        print(f"⚠️ 파일 없음: {pid} → {path}")
        continue
    print(f"▶ Processing {pid}  (Insomnia={grp})")
    mets = compute_mst_metrics(path)
    if mets is None:
        # 반환값이 None이면 헤더 실패 또는 읽기 오류이므로 건너뜀
        continue

    # 이제 안전하게 dict에 추가
    mets.update(Patient=pid, Insomnia=grp)
    metrics_list.append(mets)
    
df_metrics = pd.DataFrame(metrics_list)
df_metrics.to_csv("mst_metrics_all_subjects.csv", index=False, encoding="utf-8-sig")
print("\nMST 지표 저장됨 → mst_metrics_all_subjects.csv\n")
print(df_metrics)
