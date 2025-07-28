# =========================
# --- Глобальные настройки
# =========================
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 512
SCREEN_TITLE = "Голосомобиль"
TRACK_CENTER_Y = SCREEN_HEIGHT - 350
TRACK_HEIGHT = 150
TRACK_TOP = TRACK_CENTER_Y + TRACK_HEIGHT // 2
TRACK_BOTTOM = TRACK_CENTER_Y - TRACK_HEIGHT // 2

# Параметры движения
DURATION=2
START_X = 100
END_X = 1000
SHIP_SPEED = (END_X - START_X) / 20

# Зона pitch
PITCH_MIN = 120
PITCH_MAX = 180
SMOOTHING_FACTOR = 0.2  # сглаживание движения по Y

# Аудио
BLOCKSIZE = 1024
SILENCE_THRESHOLD_DB = 60.0
BLOCKS_TO_SILENT = 4
PITCH_FLOOR = 100
PITCH_CEILING = 600
VOICING_THRESHOLD = 0.6

# Максимальное пребывание за пределами
MAX_OUT_OF_ZONE_DURATION = 0.5