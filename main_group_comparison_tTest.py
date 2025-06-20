import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from scipy.stats import ttest_ind, mannwhitneyu
import seaborn as sns
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
from mne_connectivity import spectral_connectivity_epochs

subjects_df = pd.read_csv("C:/Users/user/Documents/Workspace/BioNet/matched_patients_eeg_insomnia_EO_amount.csv")
df_metrics = pd.read_csv("4. Beta/mst_metrics_all_subjects.csv", encoding="utf-8-sig")
# fmin=0.5
# fmax=4.0 # Delta

# fmin=4.0
# fmax=8.0, # Theta

# fmin=8.0
# fmax=13.0, # Alpha

fmin=13.0
fmax=30.0, # Beta

# ====================================================
# 4) 그룹별 통계 비교 (t-test + Mann–Whitney U)
# ====================================================
print("\n--- Group Comparison ---")
for col in ["leaf_fraction", "diameter", "mean_betweenness"]:
    ins = df_metrics.loc[df_metrics["Insomnia"]==1, col]
    noins = df_metrics.loc[df_metrics["Insomnia"]==0, col]
    t_stat, p_t = ttest_ind(ins, noins, nan_policy='omit')
    u_stat, p_u = mannwhitneyu(ins, noins, alternative='two-sided')
    print(f"{col:20s} | t-test: t={t_stat:.3f}, p={p_t:.3f}  | Mann–Whitney: U={u_stat:.1f}, p={p_u:.3f}")

    # 박스플롯
    plt.figure(figsize=(4,3))
    sns.boxplot(x="Insomnia", y=col, data=df_metrics, palette="Set2")
    plt.title(f"{col} by Group")
    plt.xlabel("Insomnia (0=no / 1=yes)")
    plt.tight_layout()
    plt.show()
    

# 1) DataFrame 준비
# 채널 이름 자동 추출 (BC_ 으로 시작하는 컬럼)
bc_cols = [c for c in df_metrics.columns if c.startswith('BC_')]

# 2) 채널별 통계 비교
sig_channels = []
p_vals = []
for ch in bc_cols:
    grp_ins = df_metrics.loc[df_metrics['Insomnia']==1, ch].dropna()
    grp_no  = df_metrics.loc[df_metrics['Insomnia']==0, ch].dropna()

    # 독립표본 t-검정
    t_stat, p_t = ttest_ind(grp_ins, grp_no, equal_var=False)
    # Mann–Whitney U 검정 (비모수)
    u_stat, p_u = mannwhitneyu(grp_ins, grp_no, alternative='two-sided')

    # 예: t-검정 p < 0.05 이면 유의하다고 본다
    if p_t < 0.05:
        sig_channels.append(ch.replace('BC_',''))
        p_vals.append(p_t)

    print(f"{ch:8s} | t-test p={p_t:.3f} | MWU p={p_u:.3f}")

print("\n유의 채널:", sig_channels)


# 3) Topomap 그리기
#  - MNE의 standard_1020 montage 사용
#  - ch_names 와 BC 값을 매핑해서 topomap 시각화

# 채널 목록과 위치 로딩
montage = mne.channels.make_standard_montage('standard_1020')
ch_names = [ch.replace('BC_','') for ch in bc_cols]
info = mne.create_info(ch_names=ch_names, sfreq=1000., ch_types='eeg')
info.set_montage(montage)

# Insomnia vs No-Insomnia 평균 BC 값 차이 topomap
# (Insomnia 평균 – No-Insomnia 평균)
data_ins = df_metrics[df_metrics['Insomnia']==1][bc_cols].mean().values
data_no  = df_metrics[df_metrics['Insomnia']==0][bc_cols].mean().values
diff     = data_ins - data_no

# MNE Evoked 객체에 담아서 topomap
evoked = mne.EvokedArray(diff[:, np.newaxis], info, tmin=0.0)

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
vmax = np.max(np.abs(diff))
mne.viz.plot_topomap(
    evoked.data[:, 0],
    evoked.info,
    axes=ax,
    show=False,
    cmap='RdBu_r',
    vlim=(-vmax, vmax)      # <-- vmin/vmax 대신 vlim
)
ax.set_title('Insomnia vs. Control (Δ mean BC)')
plt.show()


# ====================================================
# 그룹 평균 PLI 매트릭스 기반 MST 시각화
# ====================================================
def draw_group_mst(pli_group_avg, chan_names, title, fmin, fmax):
    """그룹 평균 PLI로부터 MST 그래프를 생성하고 시각화합니다."""
    pli_sym = pli_group_avg + pli_group_avg.T
    np.fill_diagonal(pli_sym, 0)
    weights = 1.0 - pli_sym
    mst_sparse = minimum_spanning_tree(weights)
    mst = mst_sparse.toarray()
    G = nx.from_numpy_array(mst)
    mapping = {i: chan_names[i] for i in range(len(chan_names))}
    G = nx.relabel_nodes(G, mapping)

    # 채널 위치
    montage = mne.channels.make_standard_montage('standard_1020')
    pos3d = montage.get_positions()['ch_pos']
    pos2d = {ch: pos3d[ch][:2] for ch in chan_names if ch in pos3d}

    plt.figure(figsize=(6,6))
    nx.draw(
        G,
        pos=pos2d,
        node_size=100,
        node_color='lightcoral',
        edge_color='gray',
        with_labels=True,
        font_size=10
    )
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
pli_all = []
pli_insomnia = []
pli_control = []
chan_names = None

for _, row in subjects_df.iterrows():
    path = row["EEG_path"]
    ins = row["Insomnia"]
    try:
        with open(path, "r") as f:
            header = [next(f) for _ in range(50)]
        for i, line in enumerate(header):
            if "CH0:FRAME" in line and "EEG" in line:
                cols = line.strip().split()
                header_idx = i
                break
        exclude = {"CH0:FRAME:N", "CH40:ECG:ECG1", "CH41:TRG:TRG"}
        eeg_cols = [c for c in cols if c not in exclude]
        df = pd.read_csv(path, delim_whitespace=True, skiprows=header_idx+1, names=cols, usecols=eeg_cols)
        chan_names = [c.split(":")[-1] for c in eeg_cols]
        df.columns = chan_names
        data = df.values.T * 0.001
        sfreq = 250
        info = mne.create_info(ch_names=chan_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        raw.set_montage(mne.channels.make_standard_montage('standard_1020'))
        raw.set_eeg_reference("average", projection=False)
        raw.crop(tmin=0.5)
        raw.filter(0.5, 40, fir_design="firwin", verbose=False)
        raw.notch_filter([60, 120], method='spectrum_fit', verbose=False)
        ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800, verbose=False)
        ica.fit(raw)
        eog_inds, _ = ica.find_bads_eog(raw, ch_name="Fp1")
        ica.exclude = eog_inds
        raw_clean = raw.copy()
        ica.apply(raw_clean)
        events = mne.make_fixed_length_events(raw_clean, id=1, duration=4.0)
        epochs = mne.Epochs(raw_clean, events, tmin=0, tmax=4, preload=True, verbose=False, baseline=(0, 0))
        data_epochs = epochs.get_data().astype(np.float64)
        conn = spectral_connectivity_epochs(
            data_epochs, names=chan_names,
            method='pli', sfreq=sfreq, mode='fourier',
            fmin=fmin, fmax=fmax, faverage=True,
            block_size=1000, n_jobs=1, verbose=False
        )
        pli = conn.get_data(output='dense').squeeze()
        pli = pli + pli.T
        np.fill_diagonal(pli, 0)
        pli_all.append(pli)
        if ins == 1:
            pli_insomnia.append(pli)
        else:
            pli_control.append(pli)
    except Exception as e:
        print(f"[건너뜀] {path}: {e}")

# 평균 PLI 계산 및 시각화
if pli_all and chan_names:
    pli_ins = np.mean(pli_insomnia, axis=0)
    pli_ctrl = np.mean(pli_control, axis=0)
    draw_group_mst(pli_ins, chan_names, "Insomnia Group MST (Theta)", fmin=fmin, fmax=fmax)
    draw_group_mst(pli_ctrl, chan_names, "Control Group MST (Theta)", fmin=fmin, fmax=fmax)
    print()