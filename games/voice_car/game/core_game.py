# games/voice_car/game/core_game.py
# -*- coding: utf-8 -*-
import arcade
import threading
import time
from pathlib import Path

from . import settings as s
from .car import Car
from .background import ScrollingBackground
from .voice import analyze_voice


class VoiceCarGame(arcade.Window):
    def __init__(self, challenge_duration=3.0, profile=None, task=None):
        super().__init__(s.SCREEN_WIDTH, s.SCREEN_HEIGHT, s.SCREEN_TITLE)

        arcade.set_background_color(arcade.color.SKY_BLUE)
        self.ship = Car()

        # Состояние голоса
        self.current_pitch = None
        self.current_voice_status = 0
        self.voice_thread_running = True

        # Параметры сессии
        self.profile = profile          # dict
        self.task = task                # dict
        self.challenge_duration = float(s.DURATION)  # раннер проставляет s.DURATION
        self.rotation_start_time = None

        # Флаги состояния
        self.started = False
        self.game_over = False
        self.success = False

        # Таймеры
        self.out_of_zone_timer = 0.0
        self.timer_start_time = None
        self.elapsed_time = 0.0

        # Фон
        self.bg_1 = arcade.load_texture("games/voice_car/components/carbg_1.png")
        self.bg_2 = arcade.load_texture("games/voice_car/components/carbg_2.png")
        self.bg_3 = arcade.load_texture("games/voice_car/components/carbg_3.png")
        scroll_speed = 800 if self.challenge_duration <= 2.5 else 500
        self.scrolling_background = ScrollingBackground(
            [self.bg_1, self.bg_2, self.bg_3],
            scroll_speed=scroll_speed
        )

        # Текст задания
        self.task_text_sprite = None
        desc = None
        if isinstance(self.task, dict):
            desc = self.task.get("description")
        else:
            desc = getattr(self.task, "description", None)
        if desc:
            font_path = str(Path("games/voice_car/components/shrift_mult.otf").absolute())
            self.task_text_sprite = arcade.create_text_sprite(
                text=desc,
                start_x=s.SCREEN_WIDTH // 2,
                start_y=s.SCREEN_HEIGHT // 2 + 100,
                font_size=50,
                font_name=font_path,
                color=arcade.color.RED_DEVIL,
                anchor_x="center",
                anchor_y="center"
            )

        # Поток анализа голоса
        threading.Thread(target=self.voice_loop, daemon=True).start()

    # ---------------- Голос ----------------
    def voice_loop(self):
        for voice_status, pitch in analyze_voice():
            if not self.voice_thread_running:
                break
            self.current_voice_status = voice_status
            self.current_pitch = pitch

    def on_close(self):
        self.voice_thread_running = False
        super().on_close()

    # ---------------- Рендер ----------------
    def on_draw(self):
        arcade.start_render()

        # 1) Фон
        self.scrolling_background.draw()

        # 2) Машина
        self.ship.draw()

        # 3) Текст задания
        if self.task_text_sprite:
            self.task_text_sprite.draw()

        # 4) Сообщения
        if self.success and not self.game_over:
            arcade.draw_text(
                "МОЛОДЕЦ!",
                s.SCREEN_WIDTH // 2, s.SCREEN_HEIGHT // 2,
                arcade.color.TEA_GREEN, 50, anchor_x="center"
            )

        if self.game_over and not self.success:
            arcade.draw_text(
                "Попробуй еще раз",
                s.SCREEN_WIDTH // 2, s.SCREEN_HEIGHT // 2,
                arcade.color.YELLOW_ORANGE, 50, anchor_x="center"
            )

        # 5) Шкала питча (по желанию)
        # self.draw_pitch_scale()

    def draw_pitch_scale(self):
        scale_x = s.SCREEN_WIDTH - 50
        top = s.SCREEN_HEIGHT - 20
        bottom = 20
        height = top - bottom

        # Ось
        arcade.draw_line(scale_x, bottom, scale_x, top, arcade.color.BLACK, 2)

        # Разметка (5 делений)
        for i in range(6):
            freq = s.PITCH_FLOOR + i * (s.PITCH_CEILING - s.PITCH_FLOOR) / 5
            y = bottom + (i / 5) * height
            arcade.draw_line(scale_x - 10, y, scale_x + 10, y, arcade.color.BLACK)
            arcade.draw_text(f"{int(freq)}", scale_x + 15, y - 8, arcade.color.BLACK, 12)

        # Плашка диапазона (s.PITCH_MIN..s.PITCH_MAX)
        y_min = bottom + ((s.PITCH_MIN - s.PITCH_FLOOR) / (s.PITCH_CEILING - s.PITCH_FLOOR)) * height
        y_max = bottom + ((s.PITCH_MAX - s.PITCH_FLOOR) / (s.PITCH_CEILING - s.PITCH_FLOOR)) * height
        arcade.draw_rectangle_filled(
            scale_x, (y_min + y_max) / 2,
            20, y_max - y_min,
            arcade.color.ALMOND
        )

        # Текущий pitch
        if self.current_pitch:
            y_current = bottom + ((self.current_pitch - s.PITCH_FLOOR) / (s.PITCH_CEILING - s.PITCH_FLOOR)) * height
            if bottom <= y_current <= top:
                arcade.draw_line(scale_x - 15, y_current, scale_x + 15, y_current, arcade.color.RED, 3)

    # ---------------- Логика ----------------
    def on_update(self, delta_time: float):
        # Конечные состояния
        if self.game_over or self.success:
            return

        voice_status = self.current_voice_status
        pitch = self.current_pitch

        # Старт игры по первому голосу
        if not self.started and voice_status == 1:
            self.started = True
            self.scrolling_background.start_challenge(duration=self.challenge_duration)

        if not self.started:
            return

        # Тикаем общий таймер
        if self.timer_start_time is None and voice_status == 1:
            self.timer_start_time = time.time()

        if self.timer_start_time and not self.game_over and self.ship.center_x < s.END_X:
            self.elapsed_time = time.time() - self.timer_start_time

        # Обновление фона (передаём факт попадания в зону)
        self.scrolling_background.update(
            delta_time,
            voice_active=(voice_status == 1),
            pitch_in_range=(pitch is not None and s.PITCH_MIN <= pitch <= s.PITCH_MAX)
        )

        # Движение/анимация машины
        self.ship.update_position(voice_status, pitch, delta_time)
        self.ship.update_rotation(delta_time)
        if not self.game_over:
            self.ship.update_animation(delta_time)

        # Успех по окончанию челленджа (если нет вращения)
        if (not self.success and not self.game_over and
                self.scrolling_background.challenge_end_time and not self.ship.rotating):
            if time.time() >= self.scrolling_background.challenge_end_time:
                self.success = True
                return

        # Проигрыш при потере голоса
        if voice_status == 0:
            self.game_over = True
            return

        # Контроль выхода за границы дорожки
        if self.ship.center_y < s.TRACK_BOTTOM or self.ship.center_y > s.TRACK_TOP:
            self.out_of_zone_timer += delta_time
            if self.out_of_zone_timer >= s.MAX_OUT_OF_ZONE_DURATION:
                if not self.ship.rotating:
                    self.ship.start_rotation()
                    self.rotation_start_time = time.time()
        else:
            self.out_of_zone_timer = 0.0

        # Завершение вращения => проигрыш
        if self.ship.rotating and not self.game_over and self.rotation_start_time:
            if time.time() - self.rotation_start_time >= 1.0:
                self.game_over = True

    # ---------------- Сброс ----------------
    def reset_game(self, task=None):
        """
        Полный сброс состояния и повторная инициализация.
        """
        self.ship = Car()
        self.current_pitch = None
        self.started = False
        self.game_over = False
        self.success = False
        self.out_of_zone_timer = 0.0
        self.timer_start_time = None
        self.elapsed_time = 0.0
        self.rotation_start_time = None

        self.scrolling_background.reset()
        self.challenge_duration = float(s.DURATION)

        if task is not None:
            self.task = task  # dict

        # Пересоберём текст задания
        desc = None
        if isinstance(self.task, dict):
            desc = self.task.get("description")
        else:
            desc = getattr(self.task, "description", None)

        if desc:
            font_path = str(Path("games/voice_car/components/shrift_mult.otf").absolute())
            self.task_text_sprite = arcade.create_text_sprite(
                text=desc,
                start_x=s.SCREEN_WIDTH // 2,
                start_y=s.SCREEN_HEIGHT // 2 + 100,
                font_size=50,
                font_name=font_path,
                color=arcade.color.RED_DEVIL,
                anchor_x="center",
                anchor_y="center"
            )
        else:
            self.task_text_sprite = None

    # ---------------- Горячая смена задания (опционально) ----------------
    def apply_task_settings(self, task, profile=None):
        """
        Опциональный метод для горячей смены задания (например, по клавише 'T').
        Нормальный поток: раннер уже проставляет s.* до создания окна.
        Здесь мы аккуратно обновим s.* из dict-структуры и синхронизируем длительность.
        """
        if profile is None:
            profile = self.profile

        # 1) Диапазон
        freq = None
        if isinstance(task, dict):
            freq = task.get("frequency")
        else:
            freq = getattr(task, "frequency", None)

        if isinstance(freq, str):
            ranges = {}
            if isinstance(profile, dict):
                ranges = (profile.get("settings", {}) or {}).get("pitch_ranges", {}) or {}
            rng = ranges.get(freq) or ranges.get(str(freq).lower())
            if rng:
                s.PITCH_MIN, s.PITCH_MAX = int(rng[0]), int(rng[1])
        elif isinstance(freq, (list, tuple)) and len(freq) == 2:
            s.PITCH_MIN, s.PITCH_MAX = int(freq[0]), int(freq[1])

        # 2) Длительность и предел выхода
        dur = task.get("duration") if isinstance(task, dict) else getattr(task, "duration", None)
        if dur is not None:
            s.DURATION = float(dur)
        moo = task.get("max_out_of_zone") if isinstance(task, dict) else getattr(task, "max_out_of_zone", None)
        if moo is not None:
            s.MAX_OUT_OF_ZONE_DURATION = float(moo)

        # 3) Производные
        s.SHIP_SPEED = (s.END_X - s.START_X) / max(s.DURATION, 0.001)

        # 4) Локальная синхронизация
        self.challenge_duration = float(s.DURATION)

        # 5) Текст задания
        desc = task.get("description") if isinstance(task, dict) else getattr(task, "description", None)
        if desc:
            font_path = str(Path("games/voice_car/components/shrift_mult.otf").absolute())
            self.task_text_sprite = arcade.create_text_sprite(
                text=desc,
                start_x=s.SCREEN_WIDTH // 2,
                start_y=s.SCREEN_HEIGHT // 2 + 100,
                font_size=50,
                font_name=font_path,
                color=arcade.color.RED_DEVIL,
                anchor_x="center",
                anchor_y="center"
            )
        else:
            self.task_text_sprite = None

    # ---------------- Клавиши ----------------
    def on_key_press(self, key, modifiers):
        # Перезапуск игры
        if key == arcade.key.R:
            self.reset_game()

        # Диалог выбора нового задания (пример)
        if key == arcade.key.T:
            try:
                from ..guui.qt_task_selector import select_task_for_profile  # относительный импорт
                profile_name = None
                if isinstance(self.profile, dict):
                    profile_name = self.profile.get("name")
                else:
                    profile_name = getattr(self.profile, "name", None)

                new_task = select_task_for_profile(profile_name) if profile_name else None
                if new_task:
                    # обновим s.* и перезапустим
                    self.apply_task_settings(new_task, self.profile)
                    self.reset_game(task=new_task)
            except Exception as e:
                print(f"[Hot task switch] error: {e}")
