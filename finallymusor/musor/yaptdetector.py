import librosa
import numpy as np

# Путь к вашему аудиофайлу
audio_file = 'silero_vad/files/femalechist2.wav'

# Загрузка аудиофайла
y, sr = librosa.load(audio_file, sr=None)  # sr=None сохраняет оригинальную частоту дискретизации

# Используем PYIN для оценки высоты тона
f0, voiced_flag, voiced_probs = librosa.pyin(
    y,
    fmin=100,  # Нижняя граница: 65 Гц
    fmax=600   # Верхняя граница: 1046 Гц
)
# Вывод результатов
print("Вычисленные значения высоты тона (в Гц):")
for i, pitch in enumerate(f0):
    if voiced_flag[i]:  # Только для кадров, где есть голос
        print(f"Кадр {i}: {pitch:.2f} Hz")
    else:
        print(f"Кадр {i}: Нет голоса")

# Опционально: Визуализация результатов
import matplotlib.pyplot as plt

times = librosa.times_like(f0)

plt.figure(figsize=(10, 6))
plt.plot(times, f0, label='F0', color='blue', linewidth=2)
plt.scatter(times[voiced_flag], f0[voiced_flag], color='green', label='Voiced')
plt.scatter(times[~voiced_flag], f0[~voiced_flag], color='red', label='Unvoiced')

plt.xlabel('Время (с)')
plt.ylabel('Частота (Гц)')
plt.title('Анализ высоты тона с помощью PYIN')
plt.legend()
plt.show()