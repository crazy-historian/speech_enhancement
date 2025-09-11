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

# ------------------------- Глобальные настройки -------------------------
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 700
SCREEN_TITLE = "Voice-Controlled Arcade Game"

SCROLL_SPEED = 25
GROUND_Y = 50
AIR_Y = 800

BLOCKSIZE = 1024
SILENCE_THRESHOLD_DB = 40.0
BLOCKS_TO_SILENT = 4
PITCH_FLOOR = 75
PITCH_CEILING = 300
VOICING_THRESHOLD = 0.6

SMOOTHING_ALPHA = 0.3
RESPONSE_FACTOR = 0.9

ARTIFACT_SCORE = 10
PITCH_TOLERANCE = 200

TEXTURE_ONE = "tekstureone.png"
TEXTURE_TWO = "teksturetoo.png"

# ------------------------- Функция GUI для настройки игры -------------------------
class GameConfigWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Настройки игры")

        layout = QVBoxLayout()
        self.inputs = {}

        # Интервалы частоты
        self.add_input_field(layout, "Нижний предел частоты (Hz):", "pitch_floor", "100")
        self.add_input_field(layout, "Верхний предел частоты (Hz):", "pitch_ceiling", "600")

        self.add_input_field(layout, "Частота задания нижняя (Hz):", "pitch_floor_task", "150")
        self.add_input_field(layout, "Частота задания верхняя (Hz):", "pitch_ceiling_task", "200")

        

        # Остальные параметры
        self.add_input_field(layout, "Частота появления заданий:", "chastota", "5")
        self.add_input_field(layout, "Длительность игры:", "game_duration", "60")
        self.add_input_field(layout, "Длительность задания:", "artifact_duration", "2.0")
        self.add_input_field(layout, "Время выполнения задания:", "check_score", "0.7")

        # Кнопка "Старт"
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
        """Сохраняет параметры и запускает игру"""
        global PITCH_FLOOR, PITCH_CEILING, PITCH_FLOOR_TASK, PITCH_CEILING_TASK
        global chastota, GAME_DURATION, ARTIFACT_DURATION, CHECK_SCORE

        PITCH_FLOOR = int(self.inputs["pitch_floor"].text())
        PITCH_CEILING = int(self.inputs["pitch_ceiling"].text())
        PITCH_FLOOR_TASK = int(self.inputs["pitch_floor_task"].text())
        PITCH_CEILING_TASK = int(self.inputs["pitch_ceiling_task"].text())


        chastota = int(self.inputs["chastota"].text())
        GAME_DURATION = int(self.inputs["game_duration"].text())

        ARTIFACT_DURATION = float(self.inputs["artifact_duration"].text())
        CHECK_SCORE = float(self.inputs["check_score"].text())

        self.close()

# ------------------------- Запуск окна перед игрой -------------------------
app = QApplication(sys.argv)
config_window = GameConfigWindow()
config_window.show()
app.exec()

# ------------------------- Функция анализа частоты и голоса -------------------------
def analyze_voice():
    with InputStream(samplerate=16000, blocksize=BLOCKSIZE, channels=1, sampwidth=2) as stream:
        stream.set_methods(UnpackRawInFloat32())
        silent_counter = 0
        start_time = time.time()
        last_valid_pitch = None  # Запоминаем последнюю частоту

        while time.time() - start_time < GAME_DURATION:
            raw_data = stream.read(BLOCKSIZE)
            if not raw_data:
                continue

            signal = stream.chain_of_methods(raw_data)
            sound = parselmouth.Sound(values=signal, sampling_frequency=stream.samplerate)

            # Интенсивность
            intensity_obj = sound.to_intensity()
            intensity_values = intensity_obj.values.T.flatten()
            avg_intensity = np.mean(intensity_values) if len(intensity_values) > 0 else -50

            # Частота (pitch)
            pitch_obj = sound.to_pitch_ac(
                time_step=0.01,
                pitch_floor=PITCH_FLOOR,
                pitch_ceiling=PITCH_CEILING,
                voicing_threshold=VOICING_THRESHOLD
            )
            pitch_values = pitch_obj.selected_array['frequency']
            pitch_values[(pitch_values == 0) | (pitch_values > PITCH_CEILING)] = np.nan
            avg_pitch = np.nanmean(pitch_values) if np.any(~np.isnan(pitch_values)) else last_valid_pitch

            # Логика определения голоса
            above_silence_threshold = avg_intensity > SILENCE_THRESHOLD_DB
            valid_pitch = avg_pitch is not None

            if above_silence_threshold and valid_pitch:
                silent_counter = 0
                last_valid_pitch = avg_pitch  # Запоминаем частоту
                yield 1, avg_pitch  # есть голос + частота
            else:
                silent_counter += 1
                if silent_counter >= BLOCKS_TO_SILENT:
                    yield 0, None  # тишина

# ------------------------- Класс персонажа -------------------------
class PlayerCharacter(arcade.Sprite):
    def __init__(self):
        super().__init__()
        self.texture = arcade.load_texture(":resources:images/animated_characters/male_person/malePerson_idle.png")
        self.scale = 0.5
        self.center_x = 150
        self.center_y = GROUND_Y
        self.target_y = GROUND_Y

    def update_position(self, pitch):
        if pitch is not None:
            target_position = GROUND_Y + ((pitch - PITCH_FLOOR) / (PITCH_CEILING - PITCH_FLOOR)) * (AIR_Y - GROUND_Y)
            self.target_y = SMOOTHING_ALPHA * target_position + (1 - SMOOTHING_ALPHA) * self.target_y

        if abs(self.center_y - self.target_y) > 20:
            self.center_y += (self.target_y - self.center_y) * RESPONSE_FACTOR
        else:
            self.center_y = self.target_y

        self.center_y = max(GROUND_Y, min(self.center_y, AIR_Y))

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
        self.current_pitch = None
        self.base_pitch=50
        self.artifact_data = []  # Список для хранения данных о частотах артефактов
        arcade.set_background_color(arcade.csscolor.SKY_BLUE)

    def setup(self):
        self.player = PlayerCharacter()
        self.artifact_list = arcade.SpriteList()
        self.score = 0
        self.start_time = time.time()
        self.voice_generator = analyze_voice()
        self.artifact_data = []  # Очистка списка артефактов
    
    def draw_pitch_scale(self):
        """Рисует частотную шкалу справа."""
        scale_x = SCREEN_WIDTH - 50  # Позиция шкалы (край экрана)
        step = 50  # Интервал по высоте в пикселях
        num_steps = 10  # Количество делений
        pitch_step = (PITCH_CEILING - PITCH_FLOOR) / num_steps  # Разница частот между делениями

        for i in range(num_steps + 1):
            freq = int(PITCH_FLOOR + i * pitch_step)
            y_pos = GROUND_Y + (i / num_steps) * (AIR_Y - GROUND_Y)

            # Рисуем горизонтальные деления шкалы
            arcade.draw_line(scale_x - 10, y_pos, scale_x + 10, y_pos, arcade.color.BLACK, 2)
            arcade.draw_text(f"{freq} Hz", scale_x + 15, y_pos - 10, arcade.color.BLACK, 14)

        # Рисуем линию, обозначающую текущую частоту персонажа
        if self.current_pitch:
            pitch_y = GROUND_Y + ((self.current_pitch - PITCH_FLOOR) / (PITCH_CEILING - PITCH_FLOOR)) * (AIR_Y - GROUND_Y)
            arcade.draw_line(scale_x - 20, pitch_y, scale_x + 20, pitch_y, arcade.color.RED, 4)


    def on_draw(self):
        arcade.start_render()
        self.artifact_list.draw()
        self.player.draw()

        elapsed = time.time() - self.start_time
        time_left = max(GAME_DURATION - elapsed, 0)

        # Вывод счета и времени
        arcade.draw_text(f"Score: {self.score}", 10, SCREEN_HEIGHT - 30, arcade.color.BLACK, 20)
        arcade.draw_text(f"Time: {int(time_left)}s", SCREEN_WIDTH - 120, SCREEN_HEIGHT - 30, arcade.color.BLACK, 20)

        # Вывод частоты последнего голоса (текущей)
        if self.current_pitch:
            arcade.draw_text(f"Pitch: {int(self.current_pitch)} Hz", SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30,
                             arcade.color.BLACK, 20, anchor_x="center")

        # Отображение частоты артефактов
        for artifact, pitch in self.artifact_data:
            if artifact.collected:
                arcade.draw_text(f"{artifact.inside_time:.1f}", artifact.center_x +300, artifact.center_y -30, arcade.color.BLACK, 14)

            arcade.draw_text(f"{int(pitch)} Hz", artifact.center_x - 10, artifact.center_y + 20,
                             arcade.color.RED, 14)

        # Завершение игры
        if time_left <= 0:
            arcade.draw_text(f"Final Score: {self.score}", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2,
                             arcade.color.RED, 40, anchor_x="center")
            arcade.draw_text("Game Over", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50,
                             arcade.color.RED, 30, anchor_x="center")
            
        self.draw_pitch_scale()

    def on_update(self, delta_time):
        elapsed = time.time() - self.start_time
        if elapsed > GAME_DURATION:
            return

        self.artifact_list.update()
        try:
            voice_status, self.current_pitch = next(self.voice_generator)
        except StopIteration:
            self.current_pitch = None
        
        if voice_status == 1:
            self.player.update_position(self.current_pitch)
        elif voice_status == 0:
            self.player.update_position(self.base_pitch)
        artifacts_to_remove = []
        for artifact, pitch in self.artifact_data:
            if artifact.check_collision(self.player, delta_time):
                artifacts_to_remove.append((artifact, pitch))
                self.artifact_list.remove(artifact)
                self.score += ARTIFACT_SCORE
        self.artifact_data = [item for item in self.artifact_data if item not in artifacts_to_remove] 

        self.spawn_artifacts()

        

    def spawn_artifacts(self):
        current_time = time.time()
        if current_time - self.last_artifact_time > chastota:
            # Выбираем случайную частоту из заданного диапазона
            random_pitch = random.uniform(PITCH_FLOOR_TASK, PITCH_CEILING_TASK)

            # Создаём новый артефакт с этой случайной частотой
            new_artifact = Artifact(random_pitch)
            self.artifact_list.append(new_artifact)
            self.artifact_data.append((new_artifact, random_pitch))

            self.last_artifact_time = current_time

class Artifact(arcade.Sprite):
    def __init__(self, pitch):

        
        super().__init__(TEXTURE_ONE, scale=0.2)
        self.center_x = SCREEN_WIDTH + random.randint(50, 300)
        self.center_y = GROUND_Y + ((pitch - PITCH_FLOOR) / (PITCH_CEILING - PITCH_FLOOR)) * (AIR_Y - GROUND_Y)
        self.change_x = -SCROLL_SPEED
        self.collected = False
        self.inside_time = 0.0
        self.last_time_inside = None
        self.pitch = pitch  # Сохраняем частоту артефакта
        self.width = SCROLL_SPEED * ARTIFACT_DURATION * 15
    
    def check_collision(self, player, delta_time):
        if arcade.check_for_collision(self, player):
            self.texture = arcade.load_texture(TEXTURE_TWO)
            self.width = SCROLL_SPEED * ARTIFACT_DURATION * 15

            if self.last_time_inside is None:
                self.last_time_inside = time.time()
            else:
                self.inside_time += delta_time

            self.collected = True
        else:
            self.texture = arcade.load_texture(TEXTURE_ONE)
            self.width = SCROLL_SPEED * ARTIFACT_DURATION * 15
            self.last_time_inside = None

        # Проверяем, находится ли персонаж на нужной частоте
        if self.inside_time >= CHECK_SCORE:
                return True  # Артефакт успешно собран

        return False



if __name__ == "__main__":
    game = VoiceArcadeGame()
    game.setup()
    arcade.run()
