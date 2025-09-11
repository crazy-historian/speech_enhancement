import arcade
from .core_game import VoiceArcadeGame
from ..guui.profiles import load_profiles  # относительный импорт из GUI


def run_slogotakt_with_task(task: dict):
    """
    Получает task-словарь, создаёт игру и запускает arcade.run().
    Ожидаемые поля в task: duration, wave_interval, wave_size,
    syllable_interval, text, mic_device_index, silence_threshold_db, profile_name.
    """
    profile_name = task.get("profile_name")
    if not profile_name:
        data = load_profiles()
        if data.get("profiles"):
            profile_name = data["profiles"][0].get("name")
            task["profile_name"] = profile_name

    mic = task.get("mic_device_index", None)
    silence_db = float(task.get("silence_threshold_db", 50.0))

    game = VoiceArcadeGame(
        task=task,
        silence_threshold_db=silence_db,
        mic_device_index=mic,
        blocks_to_silent=1,  # можно настроить
    )
    game.setup()
    arcade.run()
