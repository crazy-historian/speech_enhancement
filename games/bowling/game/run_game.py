from __future__ import annotations
import arcade
from .core_game import VoiceBowlingGame  # относительный импорт
from . import settings as s


def run_bowling_game_with_task(task: dict):
    """
    Унифицированный запуск боулинга.
    Ожидает словарь task: как минимум {"name", "duration", "text"}.
    Дополнительно можно передать mic_device_index, silence_threshold_db, profile_name.
    
    """
    s.mic_device_index = task.get("mic_device_index")
    # Если твоему движку нужны ещё поля — прокидывай их в task заранее.
    game = VoiceBowlingGame(task=task)
    game.setup()
    arcade.run()
