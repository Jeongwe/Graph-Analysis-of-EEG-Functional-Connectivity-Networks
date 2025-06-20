import pandas as pd
import matplotlib.pyplot as plt

# ================================
# 1) 파일 경로 지정
# ================================
# 본인 환경에 맞게 EEG txt 파일 경로를 지정하세요.
file_path = "C:/Users/user/Documents/Workspace/psychiatry/EEG_signal_data_250529/User/YGM 2025-02-10 10.57.04 [최연섭_V0_EC]/raw/2025-02-10 11.00.37.txt"

# ================================
# 2) 헤더(채널명) 파악
# ================================
# 파일 상위 50줄을 읽어서 채널명 라인을 찾습니다.
with open(file_path, "r") as f:
    lines = [next(f) for _ in range(50)]

header_line = None
for i, line in enumerate(lines):
    if "CH0:FRAME" in line and "EEG" in line:
        header_line = line.strip()
        header_idx = i
        break

if header_line is None:
    raise RuntimeError("채널 정보가 포함된 헤더 라인을 찾지 못했습니다.")

# 공백 단위로 분리하여 컬럼명 리스트 생성
all_cols = header_line.split()

# ================================
# 3) 필요 없는 컬럼 제거
# ================================
exclude = {"CH0:FRAME:N", "CH40:ECG:ECG1", "CH41:TRG:TRG"}
eeg_cols = [c for c in all_cols if c not in exclude]

# ================================
# 4) 데이터 프레임 로드
# ================================
# header_idx+1 줄부터 실제 숫자 데이터가 시작합니다.
df = pd.read_csv(
    file_path,
    sep="\t",
    skiprows=header_idx + 1,
    names=all_cols,
    usecols=eeg_cols
)

# ================================
# 5) 시간(Time) 컬럼 추가
# ================================
sampling_rate = 250  # Hz
df["Time"] = df.index / sampling_rate

# ================================
# 6) 채널명 간소화 (예: CH8:EEG:Fp1 → Fp1)
# ================================
df.rename(columns=lambda x: x.split(":")[-1], inplace=True)

# ================================
# 7) 원하는 채널 시각화
# ================================
plot_chs = ["Fp1", "Fpz", "Fp2"]
plt.figure(figsize=(12, 5))
for ch in plot_chs:
    if ch in df.columns:
        plt.plot(df["Time"], df[ch], label=ch)
plt.xlabel("Time (s)")
plt.ylabel("Raw amplitude (ADC counts)")
plt.title("Raw EEG signals")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
