# games/slogotakt/game/core_game.py
from typing import Optional, List

import arcade
import time
import threading

from .background import ScrollingBackground
from .player import PlayerCharacter
from .voice import analyze_voice
from .artifact import Gem
from . import settings as s
from ..guui.qt_task_selector import select_task_for_profile  # ← правильный импорт


class VoiceArcadeGame(arcade.Window):
    def __init__(self, task: dict, silence_threshold_db: float,
                 mic_device_index: Optional[int] = None,
                 blocks_to_silent: int = 1):
        super().__init__(s.SCREEN_WIDTH, s.SCREEN_HEIGHT, s.SCREEN_TITLE)

        # Профиль (для выбора задания по T)
        self.profile_name = task.get("profile_name", "Default")

        # Параметры задания
        self.game_duration = float(task["duration"])
        self.wave_interval = float(task["wave_interval"])
        self.wave_size = int(task["wave_size"])
        self.syllable_interval = float(task["syllable_interval"])
        self.current_task = task["text"]

        # Аудио
        self.silence_threshold_db = float(silence_threshold_db)
        self.mic_device_index = mic_device_index
        self.blocks_to_silent = int(blocks_to_silent)

        # Игровые сущности / состояние
        self.player: Optional[PlayerCharacter] = None
        self.gem_list: List[Gem] = []
        self.gem_count = 0
        self.wave_in_progress = False
        self.wave_collected = 0
        self.wave_failed = False
        self.score = 0
        self.max_possible_score = 0
        self.current_voice_state = 0
        self.voice_thread_running = True
        self.bg = ScrollingBackground("games/slogotakt/components/fon_bird3.png", speed=s.SCROLL_SPEED)

        self.start_time: Optional[float] = None
        self.last_wave_time: Optional[float] = None
        self.airborne_time = 0.0
        self.is_airborne = False
        self.airborne_start_time: Optional[float] = None

        self.voice_generator = None
        arcade.set_background_color(arcade.csscolor.SKY_BLUE)

    def setup(self):
        self.player = PlayerCharacter()

        self.gem_list.clear()
        self.wave_in_progress = False
        self.wave_collected = 0
        self.wave_failed = False
        self.score = 0
        self.max_possible_score = 0

        self.start_time = time.time()
        self.last_wave_time = self.start_time
        self.airborne_time = 0
        self.is_airborne = False
        self.airborne_start_time = None

        # голосовой поток с выбранным микрофоном
        self.voice_generator = analyze_voice(
            self.game_duration,
            self.silence_threshold_db,
            self.blocks_to_silent,
            mic_device_index=self.mic_device_index,
        )
        threading.Thread(target=self.voice_loop, daemon=True).start()

        self.start_new_wave()

    def voice_loop(self):
        for voice_state in self.voice_generator:
            if not self.voice_thread_running:
                break
            self.current_voice_state = voice_state

    def on_close(self):
        self.voice_thread_running = False
        try:
            arcade.unschedule(self.add_gem)
        except Exception:
            pass
        super().on_close()

    def on_draw(self):
        arcade.start_render()
        self.bg.draw()
        for gem in self.gem_list:
            gem.draw()
        if self.player:
            self.player.draw()
        if self.start_time and (time.time() - self.start_time > self.game_duration):
            arcade.draw_text("GAME OVER!", s.SCREEN_WIDTH // 2, s.SCREEN_HEIGHT // 2,
                             arcade.color.RED, 40, anchor_x="center")
            arcade.draw_text(f"Final Score: {self.score} / {self.max_possible_score}",
                             s.SCREEN_WIDTH // 2, s.SCREEN_HEIGHT // 2 - 50,
                             arcade.color.WHITE, 30, anchor_x="center")

    def on_update(self, delta_time: float):
        if self.start_time and (time.time() - self.start_time > self.game_duration):
            return

        self.bg.update()
        for gem in self.gem_list:
            gem.update()

        is_voiced = (self.current_voice_state == 1)
        if self.player:
            self.player.update_physics(is_voiced)
            self.player.update_animation(delta_time)
            if self.player.center_y > s.GROUND_Y:
                if not self.is_airborne:
                    self.is_airborne = True
                    self.airborne_start_time = time.time()
                else:
                    self.airborne_time = time.time() - (self.airborne_start_time or time.time())
            else:
                if self.is_airborne:
                    self.is_airborne = False
                    self.airborne_time = 0.0

        # столкновения
        for gem in list(self.gem_list):
            if gem.right < 10:
                gem.remove_from_sprite_lists()
                self.wave_failed = True
            if self.player and gem.check_collision_and_collect(self.player):
                if len(gem.letters) == 0:
                    self.wave_collected += 1

        self.gem_list = [g for g in self.gem_list if len(g.letters) > 0]

        # завершение волны
        if self.wave_in_progress and (self.wave_collected == self.wave_size or self.wave_failed):
            if not self.wave_failed:
                self.score += 10
            self.wave_in_progress = False
            self.last_wave_time = time.time()

        # запуск новой волны
        if (not self.wave_in_progress) and self.last_wave_time and \
           (time.time() - self.last_wave_time >= self.wave_interval):
            self.start_new_wave()

    def on_key_press(self, key, modifiers):
        if key == arcade.key.T:
            try:
                picked = select_task_for_profile(self.profile_name)
            except Exception as e:
                print(f"[TaskPicker] error: {e}")
                picked = None
            if picked:
                self.apply_task(picked)

    def apply_task(self, task: dict):
        """Применяем выбранное задание во время игры и перезапускаем волну."""
        self.wave_interval = float(task.get("wave_interval", self.wave_interval))
        self.wave_size = int(task.get("wave_size", self.wave_size))
        self.syllable_interval = float(task.get("syllable_interval", self.syllable_interval))
        self.current_task = task.get("text", self.current_task)

        # мягко перезапускаем волну под новые параметры
        try:
            arcade.unschedule(self.add_gem)
        except Exception:
            pass
        self.gem_list.clear()
        self.wave_in_progress = False
        self.wave_failed = False
        self.wave_collected = 0
        self.gem_count = 0
        self.last_wave_time = time.time() - self.wave_interval  # запустим сразу
        print(f"[Task] switched to: {self.current_task} "
              f"(wave={self.wave_size}, interval={self.wave_interval}, syllable={self.syllable_interval})")

    def start_new_wave(self):
        """Начинаем волну, добавляем гемы по таймеру с интервалом слога."""
        self.wave_in_progress = True
        self.wave_failed = False
        self.wave_collected = 0
        self.gem_count = 0
        self.max_possible_score += 10

        self.gem_list.clear()
        try:
            arcade.unschedule(self.add_gem)
        except Exception:
            pass
        arcade.schedule(self.add_gem, self.syllable_interval)

    def add_gem(self, _dt: float = 0.0):
        if self.gem_count < self.wave_size:
            gem_x = s.SCREEN_WIDTH + 100
            gem_y = s.AIR_Y
            new_gem = Gem(gem_x, gem_y, self.current_task)
            self.gem_list.append(new_gem)
            self.gem_count += 1
        else:
            arcade.unschedule(self.add_gem)
