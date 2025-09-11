# games/gromik/game/core_game.py
from __future__ import annotations

import arcade
import time
import random
import threading
import numpy as np

from . import settings as s
from .voice import analyze_voice           # ДОЛЖЕН возвращать интенсивность (dB)
from .artifact import ArtifactGroup
from .player import PlayerCharacter
from .background import ScrollingBackground


class VoiceArcadeGame(arcade.Window):
    """
    Голосовая аркада «Gromik»: управление по интенсивности (dB).
    Стиль и каркас — как в pichik (таймеры, пауза, рестарт, выбор задания).
    """

    def __init__(self):
        super().__init__(s.SCREEN_WIDTH, s.SCREEN_HEIGHT, s.SCREEN_TITLE)

        # --- базовые объекты/состояния ---
        self.player: PlayerCharacter | None = None
        self.artifact_groups: list[ArtifactGroup] = []
        self.artifact_list = arcade.SpriteList()

        self.bg = ScrollingBackground("games/gromik/components/fon_bird3.png", speed=200)

        # UI/оверлеи
        self._task_overlay = None
        self.kombo_sprite = arcade.Sprite("games/gromik/components/kombo.png", scale=0.2)
        self.kombo_sprite.center_x = s.SCREEN_WIDTH // 2
        self.kombo_sprite.center_y = s.SCREEN_HEIGHT // 2 - 200

        # --- игровые величины ---
        self.score = 0
        self.collision_counter = 0
        self.rating = 0
        self.final_text = ""
        self.total_artifacts = 0

        # --- тайминги ---
        self.start_time: float | None = None
        self.paused = False
        self.restart_requested = False
        self.game_over = False
        self.paused_time = 0.0
        self.pause_start: float | None = None

        # --- голос / интенсивность ---
        self.voice_thread_running = True
        self.voice_generator = None
        self.current_intensity: float | None = None
        self.intensity_history: list[float] = []

        # --- волны артефактов ---
        self.in_wave = False
        self.artifacts_spawned = 0
        self.artifacts_in_wave = s.ARTIFACTS_IN_WAVE
        self.artifact_interval = s.ARTIFACT_INTERVAL
        self.time_since_last_artifact = 0.0
        self.wave_timer = 0.0
        self.intensity_wave = 0.0  # требуемая интенсивность на волну

        # локальная кэш-копия частоты волн и длительности (берём из settings в setup)
        self.frequency_artifacts = s.chastota
        self.game_duration = s.GAME_DURATION

        # чтобы первая волна могла стартовать сразу
        self.wave_timer = time.time() - self.frequency_artifacts

    # -------------------- жизненный цикл --------------------

    def setup(self):
        """
        Полный сброс состояния игры + чтение актуальных параметров из settings.
        """
        # динамика параметров (могли измениться через выбор задания)
        self.frequency_artifacts = s.chastota
        self.game_duration = s.GAME_DURATION
        self.artifacts_in_wave = s.ARTIFACTS_IN_WAVE
        self.artifact_interval = s.ARTIFACT_INTERVAL

        # базовые сущности
        self.player = PlayerCharacter()
        self.artifact_groups = []
        self.artifact_list = arcade.SpriteList()

        # счёт/состояния
        self.score = 0
        self.collision_counter = 0
        self.game_over = False
        self.paused = False
        self.restart_requested = False
        self.rating = 0
        self.final_text = ""
        self.total_artifacts = 0
        self.current_intensity = None
        self.intensity_history = []

        # таймеры
        self.start_time = time.time()
        self.paused_time = 0.0
        self.pause_start = None

        # волны
        self.in_wave = False
        self.artifacts_spawned = 0
        self.time_since_last_artifact = 0.0
        self.intensity_wave = 0.0
        self.wave_timer = time.time() - self.frequency_artifacts  # стартуем сразу

        # поток голоса
        self.voice_thread_running = True
        self.voice_generator = analyze_voice()
        threading.Thread(target=self.voice_loop, daemon=True).start()

    def on_close(self):
        self.voice_thread_running = False
        super().on_close()

    # -------------------- управление --------------------

    def on_key_press(self, symbol: int, modifiers: int):
        if symbol == arcade.key.ESCAPE:
            self.paused = not self.paused
            if self.paused:
                self.pause_start = time.time()
            else:
                if self.pause_start is not None:
                    self.paused_time += time.time() - self.pause_start
                    self.pause_start = None

        elif symbol == arcade.key.R:
            self.restart_requested = True

        elif symbol == arcade.key.T:
            self.open_task_selection_dialog()

    # -------------------- голосовой поток --------------------

    def voice_loop(self):
        """
        Получает значения интенсивности (dB) из генератора analyze_voice().
        """
        for intensity in self.voice_generator:
            if not self.voice_thread_running:
                break
            self.current_intensity = intensity
            self.intensity_history.append(intensity if intensity is not None else np.nan)

    # -------------------- игровой цикл --------------------

    def on_update(self, delta_time: float):
        if self.restart_requested:
            self.voice_thread_running = False
            self.restart_requested = False
            self.game_over = False
            self.paused = False
            self.setup()
            return

        if self.game_over or self.paused:
            return

        # окончание по времени
        elapsed = time.time() - self.start_time - self.paused_time
        if elapsed >= self.game_duration:
            self._finish_game()
            return

        # обновления сцены
        self.bg.update(delta_time)
        self.player.update_animation(delta_time)
        self.player.update_position(self.current_intensity)

        self.artifact_list.update()
        self.spawn_artifacts()

        # коллизии
        for group in self.artifact_groups:
            if group.update_and_check(self.player, delta_time):
                self.score += s.ARTIFACT_SCORE

    def on_draw(self):
        arcade.start_render()
        self.bg.draw()

        for group in self.artifact_groups:
            for letter in group.letters:
                letter.draw()

        self.player.draw()
        self.draw_intensity_scale()  # отладочная шкала

        arcade.draw_text(f"Score: {self.score}", 10, s.SCREEN_HEIGHT - 30, arcade.color.BLACK, 20)

        if self.game_over:
            stars_path = f"games/gromik/components/star_{self.rating}.png"
            texture = arcade.load_texture(stars_path)
            arcade.draw_texture_rectangle(
                s.SCREEN_WIDTH // 2,
                s.SCREEN_HEIGHT // 2 - 50,
                texture.width,
                texture.height,
                texture
            )
            # self.kombo_sprite.draw()  # если надо показать комбо
        if self._task_overlay:
            self._task_overlay.draw()

    # -------------------- волны/спавн --------------------

    def spawn_artifacts(self):
        now = time.time()
        if now - self.start_time > self.game_duration:
            return

        # старт новой волны
        if (not self.in_wave) and (now - self.wave_timer >= self.frequency_artifacts):
            self.in_wave = True
            self.artifacts_spawned = 0
            self.time_since_last_artifact = now
            # для первой буквы волны выберем целевую интенсивность
            if s.selected_ranges:
                rng = random.choice(s.selected_ranges)
                # уровни могут быть нецелыми — используем uniform
                self.intensity_wave = float(random.uniform(rng[0], rng[1]))
            else:
                # запасной вариант (если не задали диапазоны)
                self.intensity_wave = 70.0

        if self.in_wave:
            if (now - self.time_since_last_artifact >= self.artifact_interval
                    and self.artifacts_spawned < self.artifacts_in_wave):

                group = ArtifactGroup(s.CURRENT_TASK_TEXT, self.intensity_wave)
                group.add_to_list(self.artifact_list)
                self.artifact_groups.append(group)

                self.artifacts_spawned += 1
                self.total_artifacts += 1
                self.time_since_last_artifact = now

            if self.artifacts_spawned >= self.artifacts_in_wave:
                self.in_wave = False
                self.wave_timer = now

    # -------------------- завершение --------------------

    def _finish_game(self):
        self.game_over = True
        self.voice_thread_running = False
        max_score = self.total_artifacts * s.ARTIFACT_SCORE
        self.rating = round((self.score / max_score) * 5) if max_score > 0 else 0
        self.rating = max(1, min(self.rating, 5))
        self.final_text = f"Игра окончена!\nВы набрали {self.score} из {max_score} очков."

    # -------------------- шкала интенсивности --------------------

    def draw_intensity_scale(self):
        scale_x = s.SCREEN_WIDTH - 50
        num_steps = 5
        # берём из settings если есть, иначе дефолты
        min_db = getattr(s, "INTENSITY_MIN_DB", 20.0)
        max_db = getattr(s, "INTENSITY_MAX_DB", 90.0)

        arcade.draw_line(scale_x, s.GROUND_Y, scale_x, s.AIR_Y, arcade.color.BLACK, 2)
        for i in range(num_steps + 1):
            val = min_db + i * (max_db - min_db) / num_steps
            y = s.GROUND_Y + (i / num_steps) * (s.AIR_Y - s.GROUND_Y)
            arcade.draw_line(scale_x - 10, y, scale_x + 10, y, arcade.color.BLACK, 2)
            arcade.draw_text(f"{int(val)} dB", scale_x + 15, y - 10, arcade.color.BLACK, 14)

        if self.current_intensity is not None:
            t = (self.current_intensity - min_db) / max(1e-6, (max_db - min_db))
            y = s.GROUND_Y + np.clip(t, 0.0, 1.0) * (s.AIR_Y - s.GROUND_Y)
            arcade.draw_line(scale_x - 20, y, scale_x + 20, y, arcade.color.RED, 4)

    # -------------------- выбор задания + перезапуск --------------------

    def restart_with_task(self, task: dict):
        """
        Применяем параметры таска к settings и просим перезапуск.
        Ожидается таск формата TaskEditorIntensity (quiet/norm/loud в dB).
        """
        # длительность/частоты/текст
        s.GAME_DURATION = int(task.get("duration", s.GAME_DURATION))
        s.ARTIFACTS_IN_WAVE = int(task.get("artifacts_count", s.ARTIFACTS_IN_WAVE))
        s.ARTIFACT_INTERVAL = float(task.get("artifact_interval", s.ARTIFACT_INTERVAL))
        s.chastota = float(task.get("frequency", getattr(s, "chastota", 6.0)))
        s.CURRENT_TASK_TEXT = task.get("text", getattr(s, "CURRENT_TASK_TEXT", "ДА"))

        # выбранные уровни (dB)
        ranges = []
        if task.get("gen_quiet"):
            ranges.append((float(task["quiet"]), float(task["quiet"])))
        if task.get("gen_norm"):
            ranges.append((float(task["norm"]), float(task["norm"])))
        if task.get("gen_loud"):
            ranges.append((float(task["loud"]), float(task["loud"])))
        if not ranges:
            ranges = [(70.0, 70.0)]
        s.selected_ranges = ranges

        # сглаживание управления (подбирай под поведение игрока)
        if task.get("smooth", True):
            s.SMOOTHING_ALPHA = 0.6
            s.RESPONSE_FACTOR = 0.9
        else:
            s.SMOOTHING_ALPHA = 0.25
            s.RESPONSE_FACTOR = 0.7

        # опционально выставим диапазон шкалы для отладки
        all_vals = [v for pair in ranges for v in pair]
        if all_vals:
            s.INTENSITY_MIN_DB = min(40.0, min(all_vals) - 10.0)
            s.INTENSITY_MAX_DB = max(90.0, max(all_vals) + 10.0)

        self.restart_requested = True
        self.paused = False

    def open_task_selection_dialog(self):
        """
        Ставит игру на паузу, открывает Qt-диалог выбора задания и перезапускает игру.
        Использует общий селектор из pichik (если он доступен в проекте).
        """
        import time as _t

        # Пауза и учёт времени
        was_paused = bool(self.paused)
        if not was_paused:
            self.paused = True
            if self.pause_start is None:
                self.pause_start = _t.time()

        # Импортируем селектор в рантайме, чтобы не ломать импорты, если модуля нет
        task = None
        try:
            try:
                # если общая утилита доступна рядом с pichik
                from ..guui.qt_task_selector import select_task_for_profile  # type: ignore
            except Exception:
                # запасной вариант: свой селектор (замени на свой путь, если используешь другой)
                from guigromik import select_task_from_file as select_task_for_profile  # type: ignore

            profile = getattr(s, "profile_name", "")
            task = select_task_for_profile(profile)
        except Exception as e:
            print("[TaskPicker] exception:", e)
            task = None

        # снимаем паузу с учётом времени
        if not was_paused and self.pause_start is not None:
            self.paused_time += _t.time() - self.pause_start
            self.pause_start = None
            self.paused = False

        if task:
            # если в таске есть метка метрики — проверим, что это интенсивность
            metric = task.get("metric")
            if (metric is None) or (metric == "intensity_db"):
                self.restart_with_task(task)
            else:
                print(f"[TaskPicker] пропущен таск с metric={metric} (ожидалось 'intensity_db').")
