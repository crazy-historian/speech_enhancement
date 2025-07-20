import arcade
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

        # Интервалы громкости
        self.add_input_field(layout, "Интервал для тихого голоса (low, high):", "low_quiet", "40")
        self.add_input_field(layout, "", "high_quiet", "45")

        self.add_input_field(layout, "Интервал для нормального голоса (low, high):", "low_norm", "45")
        self.add_input_field(layout, "", "high_norm", "50")

        self.add_input_field(layout, "Интервал для громкого голоса (low, high):", "low_loud", "60")
        self.add_input_field(layout, "", "high_loud", "65")

        # Чекбоксы для выбора диапазонов громкости
        self.use_norm = QCheckBox("Генерировать нормальную громкость")
        self.use_norm.setChecked(True)  # По умолчанию включено
        layout.addWidget(self.use_norm)

        self.use_loud = QCheckBox("Генерировать громкую громкость")
        self.use_loud.setChecked(True)  # По умолчанию включено
        layout.addWidget(self.use_loud)

        self.use_quiet = QCheckBox("Генерировать тихую громкость")
        self.use_quiet.setChecked(False)  # По умолчанию выключено
        layout.addWidget(self.use_quiet)

        # Остальные параметры
        self.add_input_field(layout, "Частота появления заданий:", "chastota", "6")
        self.add_input_field(layout, "Длительность игры:", "game_duration", "60")
        self.add_input_field(layout, "Длительность задания:", "artifact_duration", "2.0")
        self.add_input_field(layout, "Время выполнения задания:", "check_score", "0.7")

        # Кнопка "Старт"
        start_button = QPushButton("Старт")
        start_button.clicked.connect(self.start_game)
        layout.addWidget(start_button)

        self.setLayout(layout)

    def add_input_field(self, layout, label_text, key, default_value):
        """Добавляет поле ввода"""
        label = QLabel(label_text)
        layout.addWidget(label)

        input_field = QLineEdit()
        input_field.setText(default_value)
        layout.addWidget(input_field)

        self.inputs[key] = input_field

    def start_game(self):
        """Сохраняет параметры и запускает игру"""
        global low_quiet, high_quiet, low_norm, high_norm, low_loud, high_loud
        global chastota, GAME_DURATION, ARTIFACT_DURATION, CHECK_SCORE
        global selected_ranges  # Новый список с выбранными диапазонами

        low_quiet = int(self.inputs["low_quiet"].text())
        high_quiet = int(self.inputs["high_quiet"].text())

        low_norm = int(self.inputs["low_norm"].text())
        high_norm = int(self.inputs["high_norm"].text())

        low_loud = int(self.inputs["low_loud"].text())
        high_loud = int(self.inputs["high_loud"].text())

        chastota = int(self.inputs["chastota"].text())
        GAME_DURATION = int(self.inputs["game_duration"].text())

        ARTIFACT_DURATION = float(self.inputs["artifact_duration"].text())
        CHECK_SCORE = float(self.inputs["check_score"].text())

        # Заполняем список выбранных диапазонов громкости
        selected_ranges = []
        if self.use_norm.isChecked():
            selected_ranges.append((low_norm, high_norm))
        if self.use_loud.isChecked():
            selected_ranges.append((low_loud, high_loud))
        if self.use_quiet.isChecked():
            selected_ranges.append((low_quiet, high_quiet))

        self.close()


# ------------------------- Запуск окна перед игрой -------------------------
app = QApplication(sys.argv)
config_window = GameConfigWindow()
config_window.show()
app.exec()


# ------------------------- Глобальные настройки -------------------------
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 900
SCREEN_TITLE = "Voice-Controlled Arcade Game"

SCROLL_SPEED = 25
GROUND_Y = 100
AIR_Y = 800

BLOCKSIZE = 1024
SILENCE_THRESHOLD_DB = 40.0
BLOCKS_TO_SILENT = 4

SMOOTHING_ALPHA = 0.3
RESPONSE_FACTOR = 0.9

ARTIFACT_SCORE = 10

TEXTURE_ONE = "berries.png"
#TEXTURE_TWO = "teksturetoo.png"
# ------------------------- Функция анализа интенсивности голоса -------------------------
def analyze_voice():
    with InputStream(samplerate=16000, blocksize=BLOCKSIZE, channels=1, sampwidth=2) as stream:
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
            yield avg_intensity

# ------------------------- Класс персонажа -------------------------
class PlayerCharacter(arcade.Sprite):
    def __init__(self, scale=0.25):
        super().__init__()
        self.textures = []
        for i in range(1, 7):
            texture = arcade.load_texture(f"{i}_bird.png")
            self.textures.append(texture)
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
        if intensity>SILENCE_THRESHOLD_DB:
            target_position = GROUND_Y + (intensity / 100) * (AIR_Y - GROUND_Y)
            self.target_y = SMOOTHING_ALPHA * target_position + (1 - SMOOTHING_ALPHA) * self.target_y

            if abs(self.center_y - self.target_y) > 20:
                self.center_y += (self.target_y - self.center_y) * RESPONSE_FACTOR
            else:
                self.center_y = self.target_y

            self.center_y = max(GROUND_Y, min(self.center_y, AIR_Y))
        else:
            self.center_y=GROUND_Y+100

class ScrollingBackground:
    def __init__(self, texture_path, speed):
        self.speed = speed
        self.textures = [
            arcade.Sprite(texture_path, center_x=SCREEN_WIDTH // 2, center_y=SCREEN_HEIGHT // 2),
            arcade.Sprite(texture_path, center_x=SCREEN_WIDTH + SCREEN_WIDTH // 2, center_y=SCREEN_HEIGHT // 2)
        ]

    def update(self):
        for sprite in self.textures:
            sprite.center_x -= self.speed

            # Зацикливаем
            if sprite.right < 0:
                sprite.center_x += SCREEN_WIDTH * 2

    def draw(self):
        for sprite in self.textures:
            sprite.draw()

# ------------------------- Класс задания (Artifact) -------------------------
# ------------------------- Класс задания (Artifact) -------------------------
class Artifact(arcade.Sprite):
    def __init__(self, intensity):
        super().__init__(TEXTURE_ONE, scale=0.6)
        self.center_x = SCREEN_WIDTH + random.randint(50, 300)
        self.center_y = GROUND_Y + (intensity / 100) * (AIR_Y - GROUND_Y)
        self.change_x = -SCROLL_SPEED
        self.collected = False  # Флаг, был ли персонаж внутри задания
        
        self.speek_task="ПА"

        # Определение громкости и соответствующего текста
        if low_norm <= intensity <= high_norm:
            self.volume_label = "Нормально"
        elif low_loud <= intensity <= high_loud:
            self.volume_label = "Громко"
        elif low_quiet <= intensity <= high_quiet:
            self.volume_label = "Тихо"

        # Длина текстуры рассчитывается так, чтобы персонаж пролетал ее за ARTIFACT_DURATION секунд
        
    def update(self):
        self.center_x += self.change_x
        if self.right < 0:
            self.remove_from_sprite_lists()

    def check_collision(self, player):
        if arcade.check_for_collision(self, player):
            

            self.collected = True
        
        if self.inside_time>=CHECK_SCORE:
            self.remove_from_sprite_lists()
            return True  
     
        return False
    


# ------------------------- Класс игры -------------------------
class VoiceArcadeGame(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        self.player = None
        self.artifact_list = None
        self.score = 0
        self.start_time = None
        self.voice_generator = None
        self.last_artifact_time = 0
        self.current_intensity = 0
        self.total_artifacts = 0
        self.intensity_history = []  # История интенсивности
        self.artifact_intervals = []  # Хранение данных об артефактах
        self.bg = ScrollingBackground("grass_1.png", speed=20)  # заменишь на свою текстуру

    def setup(self):
        self.player = PlayerCharacter()
        self.artifact_list = arcade.SpriteList()
        self.score = 0
        self.total_artifacts = 0
        self.start_time = time.time()
        self.voice_generator = analyze_voice()
        self.intensity_history = []  # История интенсивности
        self.artifact_intervals = []  # Сброс истории артефактов
    
    def draw_intensity_scale(self):
        """Рисует шкалу громкости справа."""
        scale_x = SCREEN_WIDTH - 50  # Позиция шкалы (правый край экрана)
        num_steps = 5  # Количество делений шкалы
        min_intensity = 0  # Минимальная громкость (тишина)
        max_intensity = 100  # Максимальная громкость (очень громко)

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


    def on_draw(self):
        arcade.start_render()
        self.bg.draw()
        self.artifact_list.draw()
        self.player.draw()
        arcade.draw_text(f"Score: {self.score}", 10, SCREEN_HEIGHT - 30, arcade.color.BLACK, 20)
        elapsed = time.time() - self.start_time
        left = GAME_DURATION - elapsed
        arcade.draw_text(f"Time: {int(left)}", SCREEN_WIDTH - 120, SCREEN_HEIGHT - 30, arcade.color.BLACK, 20)
        
        if left <= 0:
            arcade.draw_text(f"Final Score: {self.score}/{self.total_artifacts * ARTIFACT_SCORE}", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, arcade.color.RED, 40, anchor_x="center")
            
            self.show_intensity_graph()

        for artifact in self.artifact_list:
            if artifact.collected:
                arcade.draw_text(f"{artifact.inside_time:.1f}", artifact.center_x - 20, artifact.center_y + 20, arcade.color.BLACK, 14)

            # Отображение громкости ("Громко" или "Нормально")
            #arcade.draw_text(artifact.volume_label, artifact.center_x - 20, artifact.center_y + 40, arcade.color.RED, 14)
            #arcade.draw_text(artifact.speek_task, artifact.center_x - 20, artifact.center_y-20, arcade.color.RED, 30)
        
        self.draw_intensity_scale()

    def on_update(self, delta_time):
        self.bg.update()
        self.player.update_animation(delta_time)
        self.player.update_position(self.current_intensity)
        elapsed = time.time() - self.start_time
        if elapsed > GAME_DURATION:
            return

        self.artifact_list.update()
        try:
            self.current_intensity = next(self.voice_generator)
        except StopIteration:
            self.current_intensity = 0

        self.intensity_history.append(self.current_intensity)  # Сохраняем значение интенсивности

        self.player.update_position(self.current_intensity)
        self.spawn_artifacts()

        for artifact in self.artifact_list:
            if artifact.check_collision(self.player):
                self.score += ARTIFACT_SCORE
                # Сохранение данных об артефакте (начало, конец, интенсивность)
                self.artifact_intervals.append((elapsed, elapsed + ARTIFACT_DURATION, artifact.center_y))


    def spawn_artifacts(self):
        current_time = time.time()
        if current_time - self.last_artifact_time > chastota:
            if selected_ranges:  # Проверяем, выбраны ли диапазоны
                intensity_range = random.choice(selected_ranges)  # Берём случайный из включенных диапазонов
                intensity = random.randint(*intensity_range)
                self.artifact_list.append(Artifact(intensity))
                self.total_artifacts += 1
                self.last_artifact_time = current_time





if __name__ == "__main__":
    game = VoiceArcadeGame()
    game.setup()
    arcade.run()