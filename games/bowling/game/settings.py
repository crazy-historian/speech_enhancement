SCREEN_WIDTH = 1020
SCREEN_HEIGHT = 512
SCREEN_TITLE = "Voice Bowling Game"

BLOCKSIZE = 1024 #размер одного блока для обработки
PITCH_FLOOR = 100 #минимальный порог частоты основного тона
PITCH_CEILING = 600 #максимальный порог частоты основного тона
SILENCE_THRESHOLD_DB = 50.0 #громкость в ДБ, с которой сигнал считается активным
VOICING_THRESHOLD = 0.6 #значение функции автокорреляции, с которого сигнал считается voiced (озвученным)
BLOCKS_TO_SILENT = 2 #кол-во блоков, отделяющее переход с voiced на unvoiced (т.е если сигнал в реальности стал unvoiced, то еще 2 блока по 1024 фрейма он считается voiced, а потом уже unvoiced)
FONT_FAMILY = "Airfool"  # поставь сюда точное family name
