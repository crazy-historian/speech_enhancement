import arcade
import time
import os
import random
import threading
import numpy as np

from game.settings import AIR_Y, ARTIFACT_SCORE, GROUND_Y, SCREEN_HEIGHT, SCREEN_WIDTH, SCREEN_TITLE, BLOCKSIZE, chastota, ARTIFACTS_IN_WAVE, ARTIFACT_INTERVAL, GAME_DURATION
from game.voice import analyze_voice
from game.artifact import ArtifactGroup
from game.player import PlayerCharacter
from game.background import ScrollingBackground
from guigromik import select_task_gui



class VoiceArcadeGame(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        self.set_update_rate(1 / 120)
        self.ctx.enable_only(self.ctx.BLEND)

        self.player = None
        self.artifact_list = None
        self.collision_counter = 0
        self.score = 0
        self.start_time = None
        self.voice_generator = None
        self.total_artifacts = 0
        self.intensity_history = []
        self.artifact_intervals = []
        self.intensity_wave = 0
        self.counter_spawn = []
        self.collision_flag = False
        self.wave_statuses = {}
        self.time_last_wave = 0
        self.game_over = False
        self.rating = 0
        self.current_intensity = 0
        self.voice_thread_running = True
        self.paused = False
        self.restart_requested = False
        self.voice_stream_ended = False
        self.artifact_groups = []
        self.selected_ranges = []

        self.task_text = "МА"
        self.chastota = 2.0
        self.game_duration = 60
        self.artifacts_in_wave = 5
        self.artifact_interval = 0.2
        self.smoothing_alpha = 0.9
        self.response_factor = 0.2

        self.kombo_sprite = arcade.Sprite("games/gromik/components/kombo.png", scale=0.2)
        self.kombo_sprite.center_x = SCREEN_WIDTH // 2
        self.kombo_sprite.center_y = SCREEN_HEIGHT // 2 - 200

        self.bg = ScrollingBackground("games/gromik/components/fon_bird3.png", speed=200)

        self.wave_timer = 0
        self.result_time = self.artifacts_in_wave * self.artifact_interval + self.chastota + 60
        self.time_since_last_artifact = 0
        self.artifacts_spawned = 0
        self.in_wave = False

        self.block_interval = BLOCKSIZE / 16000
    
    def load_task_settings(self, task):
        global GAME_DURATION, ARTIFACTS_IN_WAVE, ARTIFACT_INTERVAL, selected_ranges, CURRENT_TASK
        GAME_DURATION = task.get("duration", 60)
        ARTIFACTS_IN_WAVE = task.get("artifacts_count", 5)
        ARTIFACT_INTERVAL = task.get("artifact_interval", 0.2)
        CURRENT_TASK = task.get("text", "МА")
        selected_ranges = []
        if task.get("gen_quiet"):
            selected_ranges.append((task["quiet"], task["quiet"]))
        if task.get("gen_norm"):
            selected_ranges.append((task["norm"], task["norm"]))
        if task.get("gen_loud"):
            selected_ranges.append((task["loud"], task["loud"]))

        # настройка сглаживания
        global SMOOTHING_ALPHA, RESPONSE_FACTOR
        if task.get("smooth", True):
            SMOOTHING_ALPHA = 0.9
            RESPONSE_FACTOR = 0.2
        else:
            SMOOTHING_ALPHA = 0.25
            RESPONSE_FACTOR = 0.7

    def setup(self):
        self.player = PlayerCharacter()
        self.artifact_list = arcade.SpriteList()
        self.score = 0
        self.collision_counter = 0
        self.total_artifacts = 0
        self.counter_spawn = 0
        self.start_time = time.time()
        self.voice_generator = analyze_voice(self.game_duration)
        self.intensity_history = []
        self.artifact_intervals = []
        self.wave_timer = time.time() - self.chastota
        self.artifact_groups = []
        self.current_intensity = 0
        self.voice_thread_running = True 
        threading.Thread(target=self.voice_loop, daemon=True).start()
        threading.Timer(self.game_duration + 0.5, self.plot_intensity_graph).start()

    
    def on_key_press(self, symbol: int, modifiers: int):
        if symbol == arcade.key.ESCAPE:
            self.paused = not self.paused
        elif symbol == arcade.key.R:
            self.restart_requested = True
        elif symbol == arcade.key.T:
            self.paused = True

            def open_task_selector(_dt):
                task = select_task_gui()
                if task:
                    self.voice_thread_running = False
                    arcade.schedule_once(lambda dt: self.restart_with_task(task), 0.1)

        arcade.schedule_once(open_task_selector, 0.1)
    def restart_with_task(self, task):
        arcade.unschedule(self.restart_with_task)
        self.load_task_settings(task)
        self.setup()


    def voice_loop(self):
        for intensity in analyze_voice():
            if not self.voice_thread_running:
                break
            self.current_intensity = intensity
            self.intensity_history.append(intensity)
        self.voice_stream_ended = True
    def on_close(self):
        self.voice_thread_running = False
        super().on_close()
    
    def plot_intensity_graph(self):
        if not self.intensity_history:
            print("Нет данных для графика.")
            return

        # Используем реальный шаг времени по blocksize
        history = self.intensity_history.copy()
        timestamps = np.arange(len(history)) * self.block_interval

        # Подстраховка: обрезаем до совпадающей длины
        min_len = min(len(timestamps), len(history))
        timestamps = timestamps[:min_len]
        history = history[:min_len]

       

    def spawn_artifacts(self):
        now = time.time()
        if now - self.start_time > self.game_duration:
            return
        if not self.in_wave and now - self.wave_timer >= self.chastota:
            self.in_wave = True
            self.artifacts_spawned = 0
            self.time_since_last_artifact = now
            self.missed_in_wave = False

        if self.in_wave:
            if now - self.time_since_last_artifact >= self.artifact_interval and self.artifacts_spawned < self.artifacts_in_wave:
                if self.artifacts_spawned < 1 and self.selected_ranges:
                    intensity_range = random.choice(self.selected_ranges)
                    self.intensity_wave = random.randint(*intensity_range)

                group = ArtifactGroup(self.task_text, self.intensity_wave)
                group.add_to_list(self.artifact_list)
                self.artifact_groups.append(group)

                self.artifacts_spawned += 1
                self.total_artifacts += 1
                self.time_since_last_artifact = now

            if self.artifacts_spawned >= self.artifacts_in_wave:
                self.time_last_wave = now
                self.in_wave = False
                self.wave_timer = now
    def on_draw(self):
        arcade.start_render()
        self.bg.draw()
        for group in self.artifact_groups:
            for letter in group.letters:
                letter.draw()
        self.player.draw()
        #self.draw_intensity_scale()
        #arcade.draw_text(f"Score: {self.score}", 10, SCREEN_HEIGHT - 30, arcade.color.BLACK, 20)
        if self.game_over and self.final_text:
            arcade.draw_text(
                self.final_text,
                SCREEN_WIDTH // 2,
                SCREEN_HEIGHT // 2,
                arcade.color.BLACK,
                font_size=30,
                anchor_x="center",
                anchor_y="center"
            )
            stars_path = f"games/gromik/components/star_{self.rating}.png"
            texture = arcade.load_texture(stars_path)
            arcade.draw_texture_rectangle(
                SCREEN_WIDTH // 2,
                SCREEN_HEIGHT // 2 - 50,
                texture.width,
                texture.height,
                texture
            )
            self.kombo_sprite.draw()
        if self.paused:
            arcade.draw_text("Пауза", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 100,
                            arcade.color.GRAY, font_size=40, anchor_x="center")
    
    def update_intensity(self, delta_time):
        if self.game_over:
            return
        try:
            self.current_intensity = next(self.voice_generator)
        except StopIteration:
            self.voice_stream_ended = True
            arcade.unschedule(self.update_intensity)

    def on_update(self, delta_time):
        if self.game_over:
            return
        if self.paused:
            return

        if self.restart_requested:
            self.voice_thread_running = False
            self.restart_requested = False
            self.setup()
            return

        now = time.time()
        self.bg.update(delta_time)
        self.player.update_animation(delta_time)
        self.player.update_position(self.current_intensity)
        self.artifact_list.update()
        self.spawn_artifacts()

        for group in self.artifact_groups:
            if group.update_and_check(self.player, delta_time):
                self.score += ARTIFACT_SCORE

        if self.voice_stream_ended and not self.in_wave and time.time() - self.time_last_wave > chastota:
            self.game_over = True
            max_score = self.total_artifacts * ARTIFACT_SCORE
            self.rating = round((self.score / max_score) * 5) if max_score > 0 else 0
            self.rating = max(1, min(self.rating, 5))
            self.final_text = f"Игра окончена!\nВы набрали {self.score} из {max_score} очков."

            
    
    def draw_intensity_scale(self):
        """Рисует шкалу громкости справа."""
        scale_x = SCREEN_WIDTH - 50  # Позиция шкалы (правый край экрана)
        num_steps = 5  # Количество делений шкалы
        min_intensity = 40  # Минимальная громкость (тишина)
        max_intensity = 90  # Максимальная громкость (очень громко)

        # Рисуем ось
        arcade.draw_line(scale_x, GROUND_Y, scale_x, AIR_Y, arcade.color.BLACK, 2)

        # Рисуем деления и подписи
        for i in range(num_steps + 1):
            intensity = min_intensity + i * (max_intensity - min_intensity) / num_steps
            y_pos = GROUND_Y + (i / num_steps) * (AIR_Y - GROUND_Y)

            arcade.draw_line(scale_x - 10, y_pos, scale_x + 10, y_pos, arcade.color.BLACK, 2)
            arcade.draw_text(f"{int(intensity)} dB", scale_x + 15, y_pos - 10, arcade.color.BLACK, 14)

        # Отображаем текущий уровень громкости
        if self.current_intensity:
            intensity_y = GROUND_Y + ((self.current_intensity - min_intensity) / (max_intensity - min_intensity)) * (AIR_Y - GROUND_Y)
            arcade.draw_line(scale_x - 20, intensity_y, scale_x + 20, intensity_y, arcade.color.RED, 4)
    
    def show_final_result(self):
        max_score = self.total_artifacts * ARTIFACT_SCORE
        print(f"Игра окончена! Вы набрали {self.score} из {max_score} возможных очков.")