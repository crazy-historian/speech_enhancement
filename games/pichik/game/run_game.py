import arcade
from game.core_game import VoiceArcadeGame



def run_pitch_game_with_task(task):
    """
    Получает task-словарь из pitch_config.json, настраивает глобальные переменные,
    создаёт объект VoiceArcadeGame и запускает arcade.run().
    """
    global GAME_DURATION, ARTIFACTS_IN_WAVE, ARTIFACT_INTERVAL
    global chastota, selected_ranges, CURRENT_TASK_TEXT
    global mic_device_index, SMOOTHING_ALPHA, RESPONSE_FACTOR

    # Распакуем task
    GAME_DURATION = task.get("duration", 60)
    ARTIFACTS_IN_WAVE = float(task.get("artifacts_count", 5))
    ARTIFACT_INTERVAL = float(task.get("artifact_interval", 0.2))
    chastota = float(task.get("frequency", 6))
    CURRENT_TASK_TEXT = task.get("text", "ДА")

    # Включаем диапазоны (quiet/norm/loud) в selected_ranges
    selected_ranges = []
    if task.get("gen_quiet"):
        selected_ranges.append((task["quiet"], task["quiet"]))
    if task.get("gen_norm"):
        selected_ranges.append((task["norm"], task["norm"]))
    if task.get("gen_loud"):
        selected_ranges.append((task["loud"], task["loud"]))
    # fallback
    if not selected_ranges:
        selected_ranges = [(100, 120)]

    # Сглаживание
    if task.get("smooth", True):
        SMOOTHING_ALPHA = 0.6
        RESPONSE_FACTOR = 0.9
    else:
        SMOOTHING_ALPHA = 0.25
        RESPONSE_FACTOR = 0.7

    # Сохраним device_index, если есть
    mic_device_index = task.get("mic_device_index", None)

    # Запуск
    game = VoiceArcadeGame()
    game.setup()
    arcade.run()