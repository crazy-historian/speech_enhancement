import arcade
from game.core_game import VoiceArcadeGame

def run_game_with_task(task):
    global chastota, GAME_DURATION, ARTIFACTS_IN_WAVE, ARTIFACT_INTERVAL
    global selected_ranges, CURRENT_TASK
    global SMOOTHING_ALPHA, RESPONSE_FACTOR  # добавим их сюда

    chastota = float(task["frequency"])
    GAME_DURATION = int(task["duration"])
    ARTIFACTS_IN_WAVE = int(task["artifacts_count"])
    ARTIFACT_INTERVAL = float(task["artifact_interval"])
    CURRENT_TASK = task.get("text", "МА")

    # выбираем значения интервалов для генерации
    selected_ranges = []
    if task.get("gen_quiet"):
        selected_ranges.append((task["quiet"], task["quiet"]))
    if task.get("gen_norm"):
        selected_ranges.append((task["norm"], task["norm"]))
    if task.get("gen_loud"):
        selected_ranges.append((task["loud"], task["loud"]))

    # логика сглаживания
    if task.get("smooth", True):
        SMOOTHING_ALPHA = 0.6
        RESPONSE_FACTOR = 0.4
    else:
        SMOOTHING_ALPHA = 0.25
        RESPONSE_FACTOR = 0.7

    # запуск игры
    game = VoiceArcadeGame()
    game.chastota = chastota
    game.game_duration = GAME_DURATION
    game.artifacts_in_wave = ARTIFACTS_IN_WAVE
    game.artifact_interval = ARTIFACT_INTERVAL
    game.task_text = CURRENT_TASK
    game.selected_ranges = selected_ranges
    game.smoothing_alpha = SMOOTHING_ALPHA
    game.response_factor = RESPONSE_FACTOR

    game.setup()
    arcade.run()
