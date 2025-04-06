
import arcade
import threading
import random
import time
import numpy as np
import parselmouth
from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QSpinBox, QPushButton, QLineEdit
import sys
import os
from pathlib import Path
import json
import math


# ------------------------- Глобальные настройки -------------------------
SCREEN_WIDTH = 1280 
SCREEN_HEIGHT = 660
SCREEN_TITLE = "Voice-Controlled Arcade Game"

SCROLL_SPEED = 10
COLLECTION_TIME = 3

GROUND_Y = 100
AIR_Y = 300

CURRENT_TASK = "АА"
#дается эталонный вариант слитности, раздельности

UP_FORCE = 50
GRAVITY_FORCE = 30
MAX_UP_SPEED = 100
MAX_DOWN_SPEED = 100

WAVE_INTERVAL = 3
WAVE_SIZE = 3
ALPHABET_FOLDER = "alphabet/"

GAME_DURATION = 60

BLOCKSIZE = 1024
PITCH_FLOOR = 100
PITCH_CEILING = 600
SILENCE_THRESHOLD_DB = 40.0
VOICING_THRESHOLD = 0.6
#BLOCKS_TO_SILENT = 1


CONFIG_FILE = Path("profiles/syllable_config.json")

def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError("Config file not found.")






# ------------------------- Функция анализа голоса -------------------------
def analyze_voice(game_duration, silence_threshold_db, blocks_to_silent):
    with InputStream(samplerate=16000, blocksize=BLOCKSIZE, channels=1, sampwidth=2) as stream:
        stream.set_methods(UnpackRawInFloat32())
        silent_counter = 0
        start_time = time.time()
        while time.time() - start_time < game_duration:
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

            above_silence_threshold = avg_intensity > silence_threshold_db
            valid_pitch = (avg_pitch > PITCH_FLOOR)
            if above_silence_threshold and valid_pitch:
                silent_counter = 0
                yield 1
            else:
                silent_counter += 1
                if silent_counter >= blocks_to_silent:
                    yield 0

def load_texture_pair(filename):
    return [
        arcade.load_texture(filename),
        arcade.load_texture(filename, flipped_horizontally=True),
    ]

# ------------------------- Класс фона (ScrollingBackground) -------------------------
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

        # Устанавливаем размеры в соответствии с размером экрана
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

# ------------------------- Класс персонажа (PlayerCharacter) -------------------------
class PlayerCharacter(arcade.Sprite):
    def __init__(self):
        super().__init__()
        main_path = ":resources:images/animated_characters/male_person/malePerson"

        self.scale = 0.5
        self.character_face_direction = 0
        self.cur_texture = 0
        self.animation_timer = 0.0

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
        self.animation_timer += delta_time

        if self.center_y <= GROUND_Y + 0.1:
            if abs(self.velocity_y) < 0.1:
                if self.animation_timer >= 0.1:  # каждые 0.1 секунды (10 кадров в секунду)
                    self.animation_timer = 0.0
                    self.cur_texture = (self.cur_texture + 1) % 8
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
    Гем, состоящий из нескольких букв (sprite'ов текста на основе кастомного шрифта).
    """
    def __init__(self, x, y, word):
        self.letters = arcade.SpriteList()
        self.change_x = -SCROLL_SPEED

        font_path = str(Path("alphabet/shrift_mult.otf").absolute())  # ваш путь к OTF
        color = (165, 42, 42, 255)
        letter_spacing = 60

        for i, letter in enumerate(word):
            letter_sprite = arcade.create_text_sprite(
                text=letter,
                start_x=x + i * letter_spacing,
                start_y=y,
                font_size=48,
                font_name=font_path,
                color=color,
                anchor_x="center",
                anchor_y="center"
            )
            letter_sprite.true_x = x + i * letter_spacing  # для корректной анимации
            letter_sprite.center_y = y
            self.letters.append(letter_sprite)

    def update(self):
        for letter_sprite in self.letters:
            letter_sprite.true_x += self.change_x
            letter_sprite.center_x = round(letter_sprite.true_x)

    def draw(self):
        self.letters.draw()

    @property
    def right(self):
        if len(self.letters) == 0:
            return 0
        return max(letter.right for letter in self.letters)

    def remove_from_sprite_lists(self):
        self.letters.clear()
    
    def check_collision_and_collect(self, player):
        for letter_sprite in self.letters:
            if arcade.check_for_collision(player, letter_sprite):
                self.letters.remove(letter_sprite)  # удаляем только собранную букву
                return True  # была коллизия
        return False


# ------------------------- Класс игры (VoiceArcadeGame) -------------------------
class VoiceArcadeGame(arcade.Window):
    def __init__(self, task, silence_threshold_db):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)

        self.silence_threshold_db = silence_threshold_db
        self.game_duration = task["duration"]
        self.wave_interval = task["wave_interval"]
        self.wave_size = task["wave_size"]
        self.syllable_interval = task["syllable_interval"]
        self.current_task = task["text"]
        self.blocks_to_silent = 0


        
        self.player = None

        self.gem_list = []
        self.gem_count = 0
        

        self.wave_in_progress = False
        
        self.wave_collected = 0
        self.wave_failed = False
        self.score = 0
        self.max_possible_score = 0
        self.wave_processed = False
        self.current_voice_state = 0
        self.voice_thread_running = True
        self.bg = ScrollingBackground("fon_bird3.png", speed=10)


        self.start_time = None
        self.airborne_time = 0
        self.is_airborne = False
        self.airborne_start_time = None

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

        self.voice_generator = analyze_voice(self.game_duration, self.silence_threshold_db, self.blocks_to_silent)
        threading.Thread(target=self.voice_loop, daemon=True).start()
        self.start_new_wave()
    
    def voice_loop(self):
        for voice_state in self.voice_generator:
            if not self.voice_thread_running:
                break
            self.current_voice_state = voice_state
    def on_close(self):
        self.voice_thread_running = False
        super().on_close()

    def on_draw(self):
        arcade.start_render()
        self.bg.draw()

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

        if time.time() - self.start_time > self.game_duration:
            arcade.draw_text("GAME OVER!",
                             SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2,
                             arcade.color.RED, 40, anchor_x="center")
            arcade.draw_text(f"Final Score: {self.score} / {self.max_possible_score}",
                             SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50,
                             arcade.color.WHITE, 30, anchor_x="center")

    def on_update(self, delta_time):
        if time.time() - self.start_time > self.game_duration:
            return

        self.bg.update()

        for gem in self.gem_list:
            gem.update()

        is_voiced = (self.current_voice_state == 1)
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

        # Проверка столкновений
        for gem in self.gem_list:
            if gem.right < 10:
                gem.remove_from_sprite_lists()
                self.wave_failed = True

            if gem.check_collision_and_collect(self.player):
                if len(gem.letters) == 0:  # все буквы собраны
                    self.wave_collected += 1

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
        
        self.gem_count = 0
        self.max_possible_score += 10

        self.gem_list.clear()

        # Запускаем таймер, который будет вызывать add_gem() каждые COLLECTION_TIME / WAVE_SIZE сек.
        arcade.schedule(self.add_gem, self.syllable_interval)
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
    def run(self):
        arcade.run()

# ------------------------- Запуск -------------------------
# if __name__ == "__main__":
#     config = load_config()

#     # Берём первое задание из списка
#     task = config["tasks"][0]
#     silence_threshold_db = config["audio"].get("silence_threshold_db", 45.0)

#     game = VoiceArcadeGame(task, silence_threshold_db)
#     game.setup()
#     arcade.run()
