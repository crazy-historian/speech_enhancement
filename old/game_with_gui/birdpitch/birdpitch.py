import arcade
import threading
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

# ------------------------- Функция GUI для настройки игры -------------------------
class GameConfigWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Настройки игры")

        layout = QVBoxLayout()
        self.inputs = {}

        self.add_input_field(layout, "Интервал для тихого голоса (low, high):", "low_quiet", "45")
        self.add_input_field(layout, "", "high_quiet", "50")

        self.add_input_field(layout, "Интервал для нормального голоса (low, high):", "low_norm", "120")
        self.add_input_field(layout, "", "high_norm", "160")

        self.add_input_field(layout, "Интервал для громкого голоса (low, high):", "low_loud", "70")
        self.add_input_field(layout, "", "high_loud", "75")

        self.use_norm = QCheckBox("Генерировать нормальную громкость")
        self.use_norm.setChecked(True)
        layout.addWidget(self.use_norm)

        self.use_loud = QCheckBox("Генерировать громкую громкость")
        self.use_loud.setChecked(True)
        layout.addWidget(self.use_loud)

        self.use_quiet = QCheckBox("Генерировать тихую громкость")
        self.use_quiet.setChecked(False)
        layout.addWidget(self.use_quiet)

        self.add_input_field(layout, "Частота появления заданий:", "chastota", "6")
        self.add_input_field(layout, "Длительность игры:", "game_duration", "60")
        self.add_input_field(layout, "Кол-во артефактов в волне:", "artifacts_in_wave", "5")
        self.add_input_field(layout, "Расстояния между артефактами (с)", "artifact_interval", "0.2")

        start_button = QPushButton("Старт")
        start_button.clicked.connect(self.start_game)
        layout.addWidget(start_button)

        self.setLayout(layout)

    def add_input_field(self, layout, label_text, key, default_value):
        label = QLabel(label_text)
        layout.addWidget(label)

        input_field = QLineEdit()
        input_field.setText(default_value)
        layout.addWidget(input_field)

        self.inputs[key] = input_field

    def start_game(self):
        global low_quiet, high_quiet, low_norm, high_norm, low_loud, high_loud
        global chastota, GAME_DURATION, ARTIFACTS_IN_WAVE, ARTIFACT_INTERVAL
        global selected_ranges

        low_quiet = int(self.inputs["low_quiet"].text())
        high_quiet = int(self.inputs["high_quiet"].text())

        low_norm = int(self.inputs["low_norm"].text())
        high_norm = int(self.inputs["high_norm"].text())

        low_loud = int(self.inputs["low_loud"].text())
        high_loud = int(self.inputs["high_loud"].text())

        chastota = float(self.inputs["chastota"].text())
        GAME_DURATION = int(self.inputs["game_duration"].text())

        ARTIFACTS_IN_WAVE = float(self.inputs["artifacts_in_wave"].text())
        ARTIFACT_INTERVAL = float(self.inputs["artifact_interval"].text())

        selected_ranges = []
        if self.use_norm.isChecked():
            selected_ranges.append((low_norm, high_norm))
        if self.use_loud.isChecked():
            selected_ranges.append((low_loud, high_loud))
        if self.use_quiet.isChecked():
            selected_ranges.append((low_quiet, high_quiet))

        # Если вдруг пользователь снял все галочки, fallback
        if not selected_ranges:
            selected_ranges.append((60, 65))  # хоть что-то

        self.close()

app = QApplication(sys.argv)
config_window = GameConfigWindow()
config_window.show()
app.exec()

# ------------------------- Глобальные настройки -------------------------
SCREEN_WIDTH = 1276
SCREEN_HEIGHT = 660
SCREEN_TITLE = "Voice-Controlled Arcade Game"

SCROLL_SPEED = 15
GROUND_Y = 70
AIR_Y = 600

PITCH_FLOOR=100
PITCH_CEILING=300

BLOCKSIZE = 1024
SILENCE_THRESHOLD_DB = 50.0
BLOCKS_TO_SILENT = 8

SMOOTHING_ALPHA = 0.6
RESPONSE_FACTOR = 0.9

ARTIFACT_SCORE = 10

TEXTURE_ONE = "berries.png"

# ------------------------- Функция анализа голоса (на основе pitch) -------------------------
def analyze_voice():
    with InputStream(samplerate=16000, blocksize=BLOCKSIZE, channels=1, sampwidth=2) as stream:
        stream.set_methods(UnpackRawInFloat32())
        silent_counter = 0
        start_time = time.time()
        last_valid_pitch = None  # Запоминаем последнюю валидную частоту

        while time.time() - start_time < GAME_DURATION:
            raw_data = stream.read(BLOCKSIZE)
            if not raw_data:
                continue

            signal = stream.chain_of_methods(raw_data)
            sound = parselmouth.Sound(values=signal, sampling_frequency=stream.samplerate)

            # Интенсивность (только для определения тишины)
            intensity_obj = sound.to_intensity()
            intensity_values = intensity_obj.values.T.flatten()
            avg_intensity = np.mean(intensity_values) if len(intensity_values) > 0 else -50

            # Частота (pitch)
            pitch_obj = sound.to_pitch_ac(
                time_step=0.01,
                pitch_floor=PITCH_FLOOR,
                pitch_ceiling=PITCH_CEILING,
                voicing_threshold=0.6
            )
            pitch_values = pitch_obj.selected_array['frequency']
            # Уберём левые значения (0 или выше 300)
            pitch_values[(pitch_values == 0) | (pitch_values > 600)] = np.nan
            avg_pitch = np.nanmean(pitch_values) if np.any(~np.isnan(pitch_values)) else last_valid_pitch

            above_silence_threshold = avg_intensity > SILENCE_THRESHOLD_DB
            valid_pitch = avg_pitch is not None

            if above_silence_threshold and valid_pitch:
                silent_counter = 0
                last_valid_pitch = avg_pitch
                yield avg_pitch
            else:
                silent_counter += 1
                if silent_counter < BLOCKS_TO_SILENT and last_valid_pitch is not None:
                    yield last_valid_pitch
                else:
                    yield None
# ------------------------- Класс персонажа -------------------------
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

    def update_position(self, pitch: float):
        """Если pitch=None, опускаем птицу вниз (или берём минимально возможный)."""
        if pitch is None:
            pitch = 100  # базовый “гравитационный” pitch, чтобы птица шла вниз

        # Преобразуем pitch из диапазона [75..300] в [GROUND_Y..AIR_Y]
        target_position = GROUND_Y + ((pitch - PITCH_FLOOR) / (PITCH_CEILING - PITCH_FLOOR)) * (AIR_Y - GROUND_Y)
        self.target_y = SMOOTHING_ALPHA * target_position + (1 - SMOOTHING_ALPHA) * self.target_y

        # Плавное движение к цели
        if abs(self.center_y - self.target_y) > 20:
            self.center_y += (self.target_y - self.center_y) * RESPONSE_FACTOR
        else:
            self.center_y = self.target_y

        # Ограничение сверху/снизу
        self.center_y = max(GROUND_Y, min(self.center_y, AIR_Y))

# ------------------------- Класс фоновой прокрутки -------------------------
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

    def update(self):
        for sprite in self.textures:
            sprite.center_x -= self.speed
            if sprite.right < 0:
                sprite.center_x += SCREEN_WIDTH * 2

    def draw(self):
        for sprite in self.textures:
            sprite.draw()

# ------------------------- Класс артефакта -------------------------
class Artifact(arcade.Sprite):
    def __init__(self, intensity):
        super().__init__(TEXTURE_ONE, scale=0.4)
        self.center_x = SCREEN_WIDTH + 100
        # Преобразуем громкость (intensity) из [0..100] в [GROUND_Y..AIR_Y]
        self.center_y = GROUND_Y + ((intensity - PITCH_FLOOR) / (PITCH_CEILING - PITCH_FLOOR)) * (AIR_Y - GROUND_Y)
        self.change_x = -SCROLL_SPEED
        self.collected = False
        self.inside_time = 0.0
        self.last_time_inside = None

    def update(self):
        self.center_x += self.change_x
        if self.right < 0:
            self.remove_from_sprite_lists()

    def check_collision(self, player):
        if arcade.check_for_collision(self, player):
            self.collected = True
            self.remove_from_sprite_lists()
            return True
        return False

# ------------------------- Класс игры -------------------------
class VoiceArcadeGame(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        self.player = None
        self.artifact_list = None
        self.collision_counter = 0
        self.score = 0
        self.start_time = None
        self.voice_generator = None
        self.current_pitch = None
        self.voice_thread_running = True
        self.total_artifacts = 0
        self.intensity_history = []
        self.artifact_intervals = []
        self.intensity_wave = 0
        self.counter_spawn = []
        self.collision_flag = False
        self.wave_statuses = {}
        self.time_last_wave = 0
        self.voice_active_start = None
        self.voice_active_time = 0.0
        self.game_over = False
    
        self.bg = ScrollingBackground("fon_bird3.png", speed=10)

        self.wave_timer = 0
        self.artifacts_in_wave = ARTIFACTS_IN_WAVE
        self.artifact_interval = ARTIFACT_INTERVAL
        # result_time пригодится, если бы мы отслеживали пропуски, пока закомментируем
        self.result_time = self.artifacts_in_wave * self.artifact_interval + chastota + 60
        self.time_since_last_artifact = 0
        self.artifacts_spawned = 0
        self.in_wave = False

    def setup(self):
        self.player = PlayerCharacter()
        self.artifact_list = arcade.SpriteList()
        self.score = 0
        self.collision_counter = 0
        self.total_artifacts = 0
        self.counter_spawn = 0
        self.start_time = time.time()
        self.voice_generator = analyze_voice()
        threading.Thread(target=self.voice_loop, daemon=True).start()
        self.intensity_history = []
        self.artifact_intervals = []
        # Чтобы сразу сгенерировать волну, зададим wave_timer так,
        # что (time.time() - wave_timer) >= chastota будет верно на старте
        self.wave_timer = time.time() - chastota
    def voice_loop(self):
        for pitch in self.voice_generator:
            if not self.voice_thread_running:
                break
            self.current_pitch = pitch
    def on_close(self):
        self.voice_thread_running = False
        super().on_close()

    def spawn_artifacts(self):
        """Спавнит волну артефактов раз в chastota секунд."""
        now = time.time()
        # Если сейчас не в волне и прошло достаточно времени — запускаем новую волну
        if not self.in_wave and (now - self.wave_timer >= chastota):
            self.in_wave = True
            self.artifacts_spawned = 0
            self.time_since_last_artifact = now

        # Если мы в волне, то проверяем время, и спавним артефакты по одному через artifact_interval
        if self.in_wave:
            if (now - self.time_since_last_artifact >= self.artifact_interval
                    and self.artifacts_spawned < self.artifacts_in_wave):
                # Берём случайный диапазон из выбранных
                if self.artifacts_spawned == 0:
                    intensity_range = random.choice(selected_ranges)
                    self.intensity_wave = random.randint(*intensity_range)
                # Создаём артефакт
                self.artifact_list.append(Artifact(self.intensity_wave))
                self.artifacts_spawned += 1
                self.total_artifacts += 1
                self.time_since_last_artifact = now

            # Если нужное число артефактов уже создано, волну заканчиваем
            if self.artifacts_spawned >= self.artifacts_in_wave:
                self.in_wave = False
                self.wave_timer = now

    def on_draw(self):
        arcade.start_render()
        self.bg.draw()
        self.artifact_list.draw()
        self.player.draw()
        self.draw_pitch_scale()
        arcade.draw_text(f"Score: {self.score}", 10, SCREEN_HEIGHT - 30, arcade.color.BLACK, 20)
        if self.game_over:
            arcade.draw_text("GAME OVER!",
                            SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2,
                            arcade.color.RED, 40, anchor_x="center")
            arcade.draw_text(f"Final Score: {self.score}",
                            SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 60,
                            arcade.color.WHITE, 30, anchor_x="center")

    def on_update(self, delta_time):
        if self.game_over:
            return

        elapsed = time.time() - self.start_time
        if elapsed >= GAME_DURATION:
            self.game_over = True
            self.voice_thread_running = False
            return
        self.bg.update()
        self.player.update_animation(delta_time)

        # Получаем pitch из генератора голоса
        if self.current_pitch is not None:
            if self.voice_active_start is None:
                self.voice_active_start = time.time()
            self.voice_active_time = time.time() - self.voice_active_start
        else:
            self.voice_active_start = None
            self.voice_active_time = 0.0
        if self.current_pitch is not None:
    # если голос появился и таймер ещё не запущен — запускаем
            if self.voice_active_start is None:
                self.voice_active_start = time.time()
            # обновляем общее активное время
            self.voice_active_time = time.time() - self.voice_active_start
        else:
            # если голос пропал — сбрасываем таймер
            self.voice_active_start = None
            self.voice_active_time = 0.0

        # Обновляем позицию персонажа с учётом pitch
        self.player.update_position(self.current_pitch)

        # Двигаем артефакты и пробуем новые спавнить
        self.artifact_list.update()
        self.spawn_artifacts()

        # Проверяем коллизии с игроком
        for artifact in self.artifact_list:
            if artifact.check_collision(self.player):
                self.collision_counter += 1

        # Если собрали все артефакты из волны, дают очки
        if self.collision_counter == self.artifacts_in_wave:
            self.score += ARTIFACT_SCORE
            self.collision_counter = 0

        # Если какой-то артефакт пролетел мимо игрока
        # можно сбросить счётчик, чтобы “волна” считалась неудачной
        for artifact in self.artifact_list:
            if artifact.right < 150:
                self.collision_counter = 0
                self.collision_flag = True

    def draw_pitch_scale(self):
        """Рисует шкалу частоты справа."""
        scale_x = SCREEN_WIDTH - 50
        num_steps = 5
        min_pitch = PITCH_FLOOR
        max_pitch = PITCH_CEILING

        arcade.draw_line(scale_x, GROUND_Y, scale_x, AIR_Y, arcade.color.BLACK, 2)

        for i in range(num_steps + 1):
            pitch = int(min_pitch + i * (max_pitch - min_pitch) / num_steps)
            y_pos = GROUND_Y + (i / num_steps) * (AIR_Y - GROUND_Y)

            arcade.draw_line(scale_x - 10, y_pos, scale_x + 10, y_pos, arcade.color.BLACK, 2)
            arcade.draw_text(f"{pitch} Hz", scale_x + 15, y_pos - 10, arcade.color.BLACK, 14)

        if self.current_pitch is not None:
            # Рисуем красную поперечную линию, показывающую текущий pitch
            pitch_y = GROUND_Y + ((self.current_pitch - min_pitch) / (max_pitch - min_pitch)) * (AIR_Y - GROUND_Y)
            # ограничим от выхода за шкалу, если pitch < 75 или > 300
            pitch_y = max(GROUND_Y, min(pitch_y, AIR_Y))
            arcade.draw_line(scale_x - 20, pitch_y, scale_x + 20, pitch_y, arcade.color.RED, 4)


if __name__ == "__main__":
    game = VoiceArcadeGame()
    game.setup()
    arcade.run()
