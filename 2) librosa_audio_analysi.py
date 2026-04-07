import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

print("소리 데이터를 불러오는 중...")
d, sr = librosa.load(librosa.ex('choice'))  # d: 오디오 신호 / sr = 샘플링 레이트 (초당 샘플수)

print(f"샘플링 레이트(sr): {sr}")
print(f"데이터 길이(d): {len(d)}개")

plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
librosa.display.waveshow(d, sr=sr)  # # x축: 시간, y축: 진폭
plt.title('Visualizing Sound Variations')

S = librosa.feature.melspectrogram(y=d, sr=sr)  # 사람의 청각 특성을 반영한 Mel scale 사용
S_dB = librosa.power_to_db(S, ref=np.max)  # 데시벨 단위로 변환 -> 시각적으로 더 명확하게 보이게 하기 위함

plt.subplot(2, 1, 2)
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
plt.title('Mel-Spectrogram')
plt.colorbar(format='%+2.0f dB')

plt.tight_layout()

plt.savefig('result_plot.png')
print("분석 완료!")
plt.show()
