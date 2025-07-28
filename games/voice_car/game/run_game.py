import arcade

from game.settings import END_X, START_X
from game.core_game import VoiceCarGame


def run_game_with_profile_task(profile, task):
    global PITCH_MIN, PITCH_MAX
    global SILENCE_THRESHOLD_DB, MAX_OUT_OF_ZONE_DURATION
    global DURATION, SHIP_SPEED

    # --- подставляем диапазон частот
    if isinstance(task.frequency, str):
        pitch_range = profile.settings['pitch_ranges'].get(task.frequency)
        if pitch_range is None:
            raise ValueError(f"Не найден диапазон частот '{task.frequency}' в профиле.")
    else:
        pitch_range = task.frequency

    PITCH_MIN, PITCH_MAX = pitch_range

    # --- остальная настройка
    SILENCE_THRESHOLD_DB = profile.settings.get('volume_threshold_db', 50.0)
    MAX_OUT_OF_ZONE_DURATION = task.max_out_of_zone
    DURATION = task.duration
    SHIP_SPEED = (END_X - START_X) / DURATION

    # --- запуск игры
    game = VoiceCarGame(challenge_duration=DURATION, profile=profile, task=task)

    arcade.run()
