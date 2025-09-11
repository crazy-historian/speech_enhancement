# -*- coding: utf-8 -*-
import arcade
from .core_game import VoiceCarGame
from . import settings as s

def run_game_with_profile_task(profile: dict, task: dict):
    """
    profile: dict с ключами 'settings', 'audio', 'name'
    task:    dict с ключами 'name', 'frequency', 'duration', 'max_out_of_zone', 'description'
    """

    # --- 1) Диапазон частот ---
    freq = task.get("frequency")
    if isinstance(freq, str):
        pr = (profile.get("settings", {})
                       .get("pitch_ranges", {})
                       .get(freq) or
              profile.get("settings", {})
                     .get("pitch_ranges", {})
                     .get(str(freq).lower()))
        if not pr:
            raise ValueError(f"Не найден диапазон частот '{freq}' в profile['settings']['pitch_ranges']")
        s.PITCH_MIN, s.PITCH_MAX = int(pr[0]), int(pr[1])
    else:
        # ожидаем [min, max]
        s.PITCH_MIN, s.PITCH_MAX = int(freq[0]), int(freq[1])

    # --- 2) Порог тишины ---
    s.SILENCE_THRESHOLD_DB = float(
        profile.get("audio", {}).get(
            "silence_threshold_db",
            profile.get("settings", {}).get("volume_threshold_db", s.SILENCE_THRESHOLD_DB)
        )
    )

    # --- 3) Длительность и производные ---
    s.DURATION = float(task.get("duration", s.DURATION))
    s.MAX_OUT_OF_ZONE_DURATION = float(task.get("max_out_of_zone", s.MAX_OUT_OF_ZONE_DURATION))
    s.SHIP_SPEED = (s.END_X - s.START_X) / max(s.DURATION, 0.001)
    s.profile_name = profile.get("name", None)


    # --- 4) Запуск игры ---
    game = VoiceCarGame(
        challenge_duration=s.DURATION,
        profile=profile,
        task=task
    )
    if hasattr(game, "setup"):
        game.setup()
    arcade.run()
