import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

print("소리 데이터를 불러오는 중...")
d, sr = librosa.load(librosa.ex('choice')) 

print(f"샘플링 레이트(sr): {sr}")
print(f"데이터 길이(d): {len(d)}개")

plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
librosa.display.waveshow(d, sr=sr)
plt.title('Visualizing Sound Variations')

S = librosa.feature.melspectrogram(y=d, sr=sr)
S_dB = librosa.power_to_db(S, ref=np.max)

plt.subplot(2, 1, 2)
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
plt.title('Mel-Spectrogram')
plt.colorbar(format='%+2.0f dB')

plt.tight_layout()

plt.savefig('result_plot.png')
print("분석 완료!")
plt.show()
