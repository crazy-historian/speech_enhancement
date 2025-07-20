import parselmouth
from parselmouth.praat import call
from parselmouth import Data

# Загрузка звука
sound = parselmouth.Sound("silero_vad/files/testnaslogi2.wav")

# Вызов функции "To TextGrid (silences)"
textgrid = call(sound, "To TextGrid (silences)", 
                100, 0.0, -25.0, 0.2, 0.05, "silent", "sounding")

# Сохранение TextGrid в правильном формате
textgrid.save("output.TextGrid", format=Data.FileFormat.TEXT)
print("TextGrid успешно создан!")



