import arcade
from .core_game import VoiceArcadeGame
from . import settings as s
from ..guui.profiles import load_profiles


def run_pitch_game_with_task(task):
    
    """
    Получает task-словарь из pitch_config.json, настраивает глобальные переменные,
    создаёт объект VoiceArcadeGame и запускает arcade.run().
    """
    

    # Распакуем task
    s.GAME_DURATION = task.get("duration", 60)
    s.ARTIFACTS_IN_WAVE = float(task.get("artifacts_count", 5))
    s.ARTIFACT_INTERVAL = float(task.get("artifact_interval", 0.2))
    s.chastota = float(task.get("frequency", 6))
    s.CURRENT_TASK_TEXT = task.get("text", "ДА")
    s.SILENCE_THRESHOLD_DB = task.get("silence_threshold_db", 50.0)
    s.profile_name = task.get("profile_name", None)
    if not s.profile_name:
        data = load_profiles()
        if data.get("profiles"):
            s.profile_name = data["profiles"][0].get("name")


    # Включаем диапазоны (quiet/norm/loud) в selected_ranges
    s.selected_ranges = []
    if task.get("gen_quiet"):
        s.selected_ranges.append((task["quiet"], task["quiet"]))
    if task.get("gen_norm"):
        s.selected_ranges.append((task["norm"], task["norm"]))
    if task.get("gen_loud"):
        s.selected_ranges.append((task["loud"], task["loud"]))
    # fallback
    if not s.selected_ranges:
        s.selected_ranges = [(100, 120)]

    # Сглаживание
    if task.get("smooth", True):
        s.SMOOTHING_ALPHA = 0.6
        s.RESPONSE_FACTOR = 0.9
    else:
        s.SMOOTHING_ALPHA = 0.25
        s.RESPONSE_FACTOR = 0.7

    # Сохраним device_index, если есть
    s.mic_device_index = task.get("mic_device_index", None)

    # Запуск
    game = VoiceArcadeGame()
    game.setup()
    arcade.run()