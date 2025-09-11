import arcade
import threading
from pathlib import Path
import random
import time
import numpy as np
import parselmouth
import os
from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QCheckBox
from guiconfig import select_task_gui
from collections import deque


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
    game.setup()
    arcade.run()


SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 660
SCREEN_TITLE = "Громик/Пичик"


SCROLL_SPEED = 400
GROUND_Y = 90
AIR_Y = 450


INTENSITY_MIN = 20
INTENSITY_MAX = 90



BLOCKSIZE = 1024
SILENCE_THRESHOLD_DB = 53.0
BLOCKS_TO_SILENT = 4



ARTIFACT_SCORE = 10

TEXTURE_ONE = "berries.png"


def analyze_voice(window_size=3):
    from guiconfig import load_config
    config = load_config()
    device_index = config.get("mic_device_index")
    print(f"device_index: {device_index}")

    # Очередь для хранения последних N значений интенсивности
    smoothing_window = deque(maxlen=window_size)

    with InputStream(samplerate=16000, blocksize=BLOCKSIZE, channels=1, sampwidth=2, device=device_index) as stream:
        stream.set_methods(UnpackRawInFloat32())
        start_time = time.time()
        while time.time() - start_time < GAME_DURATION:
            raw_data = stream.read(BLOCKSIZE)
            if not raw_data:
                continue
            signal = stream.chain_of_methods(raw_data)
            sound = parselmouth.Sound(values=signal, sampling_frequency=stream.samplerate)

            intensity_obj = sound.to_intensity()
            intensity_values = intensity_obj.values.T.flatten()
            avg_intensity = np.mean(intensity_values) if len(intensity_values) > 0 else 0

            smoothing_window.append(avg_intensity)
            smoothed = np.mean(smoothing_window)
            yield smoothed


class PlayerCharacter(arcade.Sprite):
    def __init__(self, scale=0.15):
        super().__init__()
        self.textures = [arcade.load_texture(f"{i}_bird.png") for i in range(1, 7)]
        self.texture_index = 0
        self.set_texture(self.texture_index)
        
        self.scale = scale
        self.center_x = 150
        self.center_y = GROUND_Y
        self.target_y = GROUND_Y
        self.frame_timer = 0.0

    def update_animation(self, delta_time: float = 1/60):
        self.frame_timer += delta_time
        if self.frame_timer > 0.1:
            self.texture_index = (self.texture_index + 1) % len(self.textures)
            self.set_texture(self.texture_index)
            self.frame_timer = 0.0

    def update_position(self, intensity: float):
            
        
            normalized = (intensity - INTENSITY_MIN) / (INTENSITY_MAX - INTENSITY_MIN)
            normalized = max(0.0, min(normalized, 1.0))  # ограничение в пределах 0–1
        
            target_position = GROUND_Y + normalized * (AIR_Y - GROUND_Y)
            self.target_y = SMOOTHING_ALPHA * target_position + (1 - SMOOTHING_ALPHA) * self.target_y

            if abs(self.center_y - self.target_y) > 20:
                self.center_y += (self.target_y - self.center_y) * RESPONSE_FACTOR
            else:
                self.center_y = self.target_y

            self.center_y = max(GROUND_Y, min(self.center_y, AIR_Y))
       
            


class ScrollingBackground:
    def __init__(self, texture_path, speed):
        self.speed = speed
        texture = arcade.load_texture(texture_path)
        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT

        self.textures = [
            arcade.Sprite(texture_path, scale=1.0),
            arcade.Sprite(texture_path, scale=1.0)
        ]

        for sprite in self.textures:
            sprite.width = self.width
            sprite.height = self.height

        self.textures[0].center_x = SCREEN_WIDTH // 2
        self.textures[0].center_y = SCREEN_HEIGHT // 2

        self.textures[1].center_x = SCREEN_WIDTH + SCREEN_WIDTH // 2
        self.textures[1].center_y = SCREEN_HEIGHT // 2

        # Добавим внутренние координаты для интерполяции
        self.true_x = [float(sprite.center_x) for sprite in self.textures]

    def update(self, delta_time):
        for i, sprite in enumerate(self.textures):
            self.true_x[i] -= self.speed * delta_time
            sprite.center_x = round(self.true_x[i])
            if sprite.right < 0:
                self.true_x[i] += SCREEN_WIDTH * 2
                sprite.center_x = round(self.true_x[i])

    def draw(self):
        for sprite in self.textures:
            sprite.draw()



    
class ArtifactGroup:
    def __init__(self, task, intensity):
        self.letters = []
        spacing = 5  # расстояние между буквами
        font_path = str(Path("alphabet/shrift_mult.otf").absolute())
        color=(165, 42, 42, 255)  # мягкий чёрный

        normalized = (intensity - INTENSITY_MIN) / (INTENSITY_MAX - INTENSITY_MIN)
        normalized = max(0.0, min(normalized, 1.0))
        center_y = GROUND_Y + normalized * (AIR_Y - GROUND_Y)

        current_x = SCREEN_WIDTH + 100

        for letter in task:
            sprite = arcade.create_text_sprite(
                text=letter,
                start_x=current_x,
                start_y=center_y,
                font_size=48,
                font_name=font_path,
                color=color,
                anchor_x="center",
                anchor_y="center"
            )
            sprite.true_x = current_x
            sprite.center_y = center_y
            sprite.collected = False

            self.letters.append(sprite)
            current_x += sprite.width + spacing  # расстояние зависит от ширины


        self.collected = False

    def add_to_list(self, sprite_list):
        for letter in self.letters:
            sprite_list.append(letter)

    def update_and_check(self, player, delta_time):
        if self.collected:
            return False
        all_collected = True
        for sprite in self.letters:
            if sprite.collected:
                continue

            sprite.true_x -= SCROLL_SPEED * delta_time
            sprite.center_x = round(sprite.true_x)

            dx = abs(sprite.center_x - player.center_x)
            dy = abs(sprite.center_y - player.center_y)
            if dx <= sprite.width // 2+20 and dy <= sprite.height // 2+20:
                sprite.collected = True
                sprite.remove_from_sprite_lists()
            else:
                all_collected = False

        if all_collected:
            self.collected = True
            return True
        return False


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
        self.kombo_sprite = arcade.Sprite("stars/kombo.png", scale=0.2)
        self.kombo_sprite.center_x = SCREEN_WIDTH // 2
        self.kombo_sprite.center_y = SCREEN_HEIGHT // 2 - 200
        self.current_intensity = 0
        self.voice_thread_running = True
        self.paused = False
        self.restart_requested = False

        self.artifact_groups = []
        self.voice_stream_ended = False

        self.bg = ScrollingBackground("fon_bird3.png", speed=200)

        self.wave_timer = 0
        self.artifacts_in_wave = ARTIFACTS_IN_WAVE
        self.artifact_interval = ARTIFACT_INTERVAL
        self.result_time = self.artifacts_in_wave * self.artifact_interval + chastota + 60
        self.time_since_last_artifact = 0
        self.artifacts_spawned = 0
        self.in_wave = False

        self.block_interval = BLOCKSIZE / 16000  # ~0.064s
    
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
        self.voice_generator = analyze_voice()
        self.intensity_history = []
        self.artifact_intervals = []
        self.wave_timer = time.time() - chastota
        self.artifact_groups = []
        self.current_intensity = 0
        self.voice_thread_running = True 
        threading.Thread(target=self.voice_loop, daemon=True).start()
        threading.Timer(GAME_DURATION + 0.5, self.plot_intensity_graph).start()

    
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

        # Рисуем график
        plt.figure(figsize=(10, 4))
        plt.plot(timestamps, history, color='orange', label='Интенсивность')
        plt.title("График интенсивности за всё время игры")
        plt.xlabel("Время (секунды)")
        plt.ylabel("Интенсивность, dB")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Сохраняем с timestamp в имя
        filename = f"intensity_{time.strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename)
        plt.close()

        # Открываем через системную утилиту (macOS)
        os.system(f"open {filename}")

    def spawn_artifacts(self):
        now = time.time()
        # запрет генерации новых волн после истечения времени
        if now - self.start_time > GAME_DURATION:
            return
        if not self.in_wave and now - self.wave_timer >= chastota:
            self.in_wave = True
            self.artifacts_spawned = 0
            self.time_since_last_artifact = now
            self.missed_in_wave = False

        if self.in_wave:
            if now - self.time_since_last_artifact >= self.artifact_interval and self.artifacts_spawned < self.artifacts_in_wave:
                if self.artifacts_spawned < 1:
                    intensity_range = random.choice(selected_ranges)
                    self.intensity_wave = random.randint(*intensity_range)

                # создаём группу артефактов из букв слога
                group = ArtifactGroup(CURRENT_TASK, self.intensity_wave)
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
            stars_path = f"stars/star_{self.rating}.png"
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
            self.plot_after_exit()
            arcade.exit()  # корректное завершение окна
            max_score = self.total_artifacts * ARTIFACT_SCORE
            #self.rating = round((self.score / max_score) * 5) if max_score > 0 else 0
            #self.rating = max(1, min(self.rating, 5))
            #self.final_text = f"Игра окончена!\nВы набрали {self.score} из {max_score} очков."

            
    
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




if __name__ == "__main__":
    game = VoiceArcadeGame()
    game.setup()
    arcade.run()

