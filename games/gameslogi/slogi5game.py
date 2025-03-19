import arcade
import random
import time
import numpy as np
import parselmouth
from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QSpinBox, QPushButton, QLineEdit
import sys
import os

# ------------------------- Глобальные настройки -------------------------
SCREEN_WIDTH = 1020
SCREEN_HEIGHT = 512
SCREEN_TITLE = "Voice-Controlled Arcade Game"

SCROLL_SPEED = 25
COLLECTION_TIME = 3

GROUND_Y = 100
AIR_Y = 300

CURRENT_TASK = "МА"

UP_FORCE = 50
GRAVITY_FORCE = 30
MAX_UP_SPEED = 100
MAX_DOWN_SPEED = 100

WAVE_INTERVAL = 3
WAVE_SIZE = 3
ALPHABET_FOLDER = "alphabet40/"

GAME_DURATION = 60

BLOCKSIZE = 1024
PITCH_FLOOR = 100
PITCH_CEILING = 600
SILENCE_THRESHOLD_DB = 40.0
VOICING_THRESHOLD = 0.6
BLOCKS_TO_SILENT = 2

# ---------------------- Класс GUI ----------------------
class GameConfigWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Настройки игры")
        self.setGeometry(100, 100, 300, 200)

        layout = QVBoxLayout()
        self.inputs = {}

        # Поля ввода
        self.add_input_field(layout, "Интервал между волнами (сек):", "wave_interval", str(WAVE_INTERVAL))
        self.add_input_field(layout, "Количество камней в волне:", "wave_size", str(WAVE_SIZE))
        self.add_input_field(layout, "Длительность игры (сек):", "game_duration", str(GAME_DURATION))
        self.add_input_field(layout, "Длительность волны (сек):", "collection_time", str(COLLECTION_TIME))
        self.add_input_field(layout, "Текущие задание:", "current_task", str(CURRENT_TASK))

        # Кнопка "Старт"
        start_button = QPushButton("Запустить игру")
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
        global WAVE_INTERVAL, WAVE_SIZE, GAME_DURATION, COLLECTION_TIME, CURRENT_TASK
        WAVE_INTERVAL = int(self.inputs["wave_interval"].text())
        WAVE_SIZE = int(self.inputs["wave_size"].text())
        GAME_DURATION = int(self.inputs["game_duration"].text())
        COLLECTION_TIME = int(self.inputs["collection_time"].text())
        CURRENT_TASK = str(self.inputs["current_task"].text())

        self.close()

# ---------------------- Запуск окна перед игрой ----------------------
app = QApplication(sys.argv)
config_window = GameConfigWindow()
config_window.show()
app.exec()

# ------------------------- Функция анализа голоса -------------------------
def analyze_voice():
    with InputStream(samplerate=16000, blocksize=BLOCKSIZE, channels=1, sampwidth=2) as stream:
        stream.set_methods(UnpackRawInFloat32())
        silent_counter = 0
        start_time = time.time()
        while time.time() - start_time < GAME_DURATION:
            raw_data = stream.read(BLOCKSIZE)
            if not raw_data:
                continue

            signal = stream.chain_of_methods(raw_data)
            sound = parselmouth.Sound(values=signal, sampling_frequency=stream.samplerate)

            intensity_obj = sound.to_intensity()
            intensity_values = intensity_obj.values.T.flatten()
            avg_intensity = np.mean(intensity_values) if len(intensity_values) > 0 else -50

            pitch_obj = sound.to_pitch_ac(
                time_step=0.01,
                pitch_floor=PITCH_FLOOR,
                pitch_ceiling=PITCH_CEILING,
                voicing_threshold=VOICING_THRESHOLD,
                octave_cost=0.01,
                octave_jump_cost=0.35,
                voiced_unvoiced_cost=0.14
            )
            pitch_values = pitch_obj.selected_array['frequency']
            pitch_values[(pitch_values == 0) | (pitch_values > PITCH_CEILING)] = np.nan
            avg_pitch = np.nanmean(pitch_values) if np.any(~np.isnan(pitch_values)) else np.nan

            above_silence_threshold = avg_intensity > SILENCE_THRESHOLD_DB
            valid_pitch = (avg_pitch > PITCH_FLOOR)
            if above_silence_threshold and valid_pitch:
                silent_counter = 0
                yield 1
            else:
                silent_counter += 1
                if silent_counter >= BLOCKS_TO_SILENT:
                    yield 0

def load_texture_pair(filename):
    return [
        arcade.load_texture(filename),
        arcade.load_texture(filename, flipped_horizontally=True),
    ]

# ------------------------- Класс фона (ScrollingBackground) -------------------------
class ScrollingBackground:
    def __init__(self, width, height, image_name, scroll_speed):
        self.width = width
        self.height = height
        self.scroll_speed = scroll_speed
        self.background_list = arcade.SpriteList()

        self.sprite_1 = arcade.Sprite(image_name, scale=1.0)
        self.sprite_2 = arcade.Sprite(image_name, scale=1.0)

        self.sprite_1.center_x = self.width // 2
        self.sprite_1.center_y = self.height // 2
        self.sprite_2.center_x = self.width + self.width // 2
        self.sprite_2.center_y = self.height // 2

        self.background_list.append(self.sprite_1)
        self.background_list.append(self.sprite_2)

    def update(self):
        for sprite in self.background_list:
            sprite.center_x -= self.scroll_speed
            if sprite.right < 0:
                sprite.left = self.width

    def draw(self):
        self.background_list.draw()

# ------------------------- Класс персонажа (PlayerCharacter) -------------------------
class PlayerCharacter(arcade.Sprite):
    def __init__(self):
        super().__init__()
        main_path = ":resources:images/animated_characters/male_person/malePerson"

        self.scale = 0.5
        self.character_face_direction = 0
        self.cur_texture = 0

        self.idle_texture_pair = load_texture_pair(f"{main_path}_idle.png")
        self.jump_texture_pair = load_texture_pair(f"{main_path}_jump.png")
        self.fall_texture_pair = load_texture_pair(f"{main_path}_fall.png")

        self.walk_textures = []
        for i in range(8):
            texture = load_texture_pair(f"{main_path}_walk{i}.png")
            self.walk_textures.append(texture)

        self.texture = self.idle_texture_pair[self.character_face_direction]
        self.center_x = 150
        self.center_y = GROUND_Y

        self.hit_box = self.texture.hit_box_points
        self.velocity_y = 0.0

    def update_physics(self, is_voiced: bool):
        if is_voiced:
            self.velocity_y += UP_FORCE
            if self.velocity_y > MAX_UP_SPEED:
                self.velocity_y = MAX_UP_SPEED
        else:
            self.velocity_y -= GRAVITY_FORCE
            if self.velocity_y < -MAX_DOWN_SPEED:
                self.velocity_y = -MAX_DOWN_SPEED

        self.center_y += self.velocity_y

        if self.center_y < GROUND_Y:
            self.center_y = GROUND_Y
            self.velocity_y = 0
        elif self.center_y > AIR_Y:
            self.center_y = AIR_Y
            self.velocity_y = 0

    def update_animation(self, delta_time: float = 1/60):
        if self.center_y <= GROUND_Y + 0.1:
            if abs(self.velocity_y) < 0.1:
                self.cur_texture += 1
                if self.cur_texture >= 8:
                    self.cur_texture = 0
                self.texture = self.walk_textures[self.cur_texture][self.character_face_direction]
            else:
                self.texture = self.walk_textures[self.cur_texture][self.character_face_direction]
        elif self.center_y >= AIR_Y - 0.1:
            if abs(self.velocity_y) < 0.1:
                self.texture = self.idle_texture_pair[self.character_face_direction]
            elif self.velocity_y > 0:
                self.texture = self.jump_texture_pair[self.character_face_direction]
            else:
                self.texture = self.fall_texture_pair[self.character_face_direction]
        else:
            if self.velocity_y > 0:
                self.texture = self.jump_texture_pair[self.character_face_direction]
            else:
                self.texture = self.fall_texture_pair[self.character_face_direction]

# ------------------------- Класс «гема» (Gem) -------------------------
class Gem:
    """
    Гем, состоящий из нескольких букв (спрайтов). Например, 'ПА'.
    """
    def __init__(self, x, y, word):
        self.letters = arcade.SpriteList()
        self.change_x = -SCROLL_SPEED
        self.is_collected = False  # Флаг, собран ли гем персонажем

        letter_spacing = 70
        for i, letter in enumerate(word):
            path = os.path.join(ALPHABET_FOLDER, f"{letter} beresta.jpg")
            if not os.path.exists(path):
                print(f"Файл не найден: {path}")
                continue
            letter_sprite = arcade.Sprite(path, scale=0.2)
            letter_sprite.center_x = x + i * letter_spacing
            letter_sprite.center_y = y
            self.letters.append(letter_sprite)

    def update(self):
        """Сдвигаем буквы гема влево."""
        for letter_sprite in self.letters:
            letter_sprite.center_x += self.change_x

    def draw(self):
        self.letters.draw()

    @property
    def right(self):
        """Возвращаем правую границу гема (макс. x среди всех букв)."""
        if len(self.letters) == 0:
            return 0
        return max(letter.right for letter in self.letters)

    def remove_from_sprite_lists(self):
        """Удаляем все буквы гема из SpriteList."""
        self.letters.clear()

# ------------------------- Класс игры (VoiceArcadeGame) -------------------------
class VoiceArcadeGame(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        self.background = None
        self.player = None

        self.gem_list = []
        self.gem_count = 0
        self.current_task = CURRENT_TASK  # Слово на геме

        self.wave_in_progress = False
        self.wave_size = 0
        self.wave_collected = 0
        self.wave_failed = False
        self.score = 0
        self.max_possible_score = 0
        self.wave_processed = False

        self.start_time = None
        self.airborne_time = 0
        self.is_airborne = False
        self.airborne_start_time = None

        self.voice_generator = None
        arcade.set_background_color(arcade.csscolor.SKY_BLUE)

    def setup(self):
        self.background = ScrollingBackground(
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            ":resources:images/backgrounds/abstract_1.jpg",
            SCROLL_SPEED
        )
        self.player = PlayerCharacter()

        self.gem_list.clear()
        self.wave_in_progress = False
        self.wave_size = 0
        self.wave_collected = 0
        self.wave_failed = False
        self.score = 0
        self.max_possible_score = 0

        self.start_time = time.time()
        self.last_wave_time = self.start_time
        self.airborne_time = 0
        self.is_airborne = False
        self.airborne_start_time = None

        self.voice_generator = analyze_voice()
        self.start_new_wave()

    def on_draw(self):
        arcade.start_render()
        self.background.draw()

        for gem in self.gem_list:
            gem.draw()

        self.player.draw()

        arcade.draw_text(f"Score: {self.score}",
                         10, SCREEN_HEIGHT - 30,
                         arcade.color.BLACK, 20)
        arcade.draw_text(f"Time left: {int(GAME_DURATION - (time.time() - self.start_time))}",
                         SCREEN_WIDTH - 120, SCREEN_HEIGHT - 30,
                         arcade.color.BLACK, 20)

        arcade.draw_text(f"Air time: {self.airborne_time:.1f}s",
                         SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30,
                         arcade.color.BLACK, 20, anchor_x="center")

        if time.time() - self.start_time > GAME_DURATION:
            arcade.draw_text("GAME OVER!",
                             SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2,
                             arcade.color.RED, 40, anchor_x="center")
            arcade.draw_text(f"Final Score: {self.score} / {self.max_possible_score}",
                             SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50,
                             arcade.color.WHITE, 30, anchor_x="center")

    def on_update(self, delta_time):
        if time.time() - self.start_time > GAME_DURATION:
            return

        self.background.update()

        for gem in self.gem_list:
            gem.update()

        try:
            voiced_state = next(self.voice_generator)
        except StopIteration:
            voiced_state = 0

        is_voiced = (voiced_state == 1)
        self.player.update_physics(is_voiced)

        # Считаем таймер полета
        if self.player.center_y > GROUND_Y:
            if not self.is_airborne:
                self.is_airborne = True
                self.airborne_start_time = time.time()
            else:
                self.airborne_time = time.time() - self.airborne_start_time
        else:
            if self.is_airborne:
                self.is_airborne = False
                self.airborne_time = 0

        self.player.update_animation(delta_time)

        # Проверка столкновений: удаляем буквы по отдельности
        for gem in self.gem_list:
            # Если гем полностью ушёл за левый край, и в нем нет букв, удаляем его.
            if gem.right < 0:
                if len(gem.letters) == 0:
                    gem.remove_from_sprite_lists()
                else:
                    # Если гем ушел за экран, но в нем еще остались буквы,
                    # оставляем гем, чтобы игрок мог их собрать.
                    pass

            # Проверяем столкновение с каждой буквой
            letters_to_remove = []
            for letter_sprite in gem.letters:
                if arcade.check_for_collision(self.player, letter_sprite):
                    letters_to_remove.append(letter_sprite)
                    self.wave_collected += 1
            # Удаляем собранные буквы
            for letter in letters_to_remove:
                gem.letters.remove(letter)

        # Удаляем гемы, если в них не осталось букв
        self.gem_list = [g for g in self.gem_list if len(g.letters) > 0]

        # Завершение волны
        if self.wave_in_progress and (self.wave_collected == self.wave_size or self.wave_failed):
            if not self.wave_failed:
                self.score += 10
            self.wave_in_progress = False
            self.last_wave_time = time.time()

        # Запуск новой волны через WAVE_INTERVAL
        if not self.wave_in_progress and (time.time() - self.last_wave_time >= WAVE_INTERVAL):
            self.start_new_wave()
        

    # ---------------------- САМАЯ ГЛАВНАЯ ЧАСТЬ: добавляем гемы по таймеру ----------------------
    def start_new_wave(self):
        """Запускаем волну и добавляем гемы один за другим через равные промежутки."""
        self.wave_in_progress = True
        self.wave_failed = False
        self.wave_collected = 0
        self.wave_size = WAVE_SIZE
        self.gem_count = 0
        self.max_possible_score += 10

        self.gem_list.clear()

        # Запускаем таймер, который будет вызывать add_gem() каждые COLLECTION_TIME / WAVE_SIZE сек.
        arcade.schedule(self.add_gem, COLLECTION_TIME / self.wave_size)
        print("Началась новая волна!")

    def add_gem(self, delta_time=0):
        """Добавляет один гем (слово current_task) за каждый вызов schedule."""
        if self.gem_count < self.wave_size:
            gem_x = SCREEN_WIDTH + 100
            gem_y = AIR_Y
            new_gem = Gem(gem_x, gem_y, self.current_task)
            self.gem_list.append(new_gem)
            self.gem_count += 1
            print(f"Добавлен гем {self.gem_count}/{self.wave_size}")
        else:
            # Если все гемы добавлены, останавливаем таймер
            arcade.unschedule(self.add_gem)

# ------------------------- Запуск -------------------------
if __name__ == "__main__":
    game = VoiceArcadeGame()
    game.setup()
    arcade.run()
