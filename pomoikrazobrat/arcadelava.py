# Игровой код на Arcade, где:
# 1) Вместо сбора колец появляются препятствия ("горы с лавой").
# 2) Цель: не столкнуться с препятствием.
# 3) Очки копятся сами по себе (скажем, +1 каждый кадр или +10 в секунду).
# 4) Со временем скорость игры (SCROLL_SPEED) растёт.
# 5) Логика слитной речи: персонаж плавно взлетает, иначе падает (мини-физика).
# 6) При столкновении с препятствием — game over.

import arcade
import random
import time
import numpy as np
import parselmouth
from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32

# ------------------------- Глобальные настройки -------------------------
SCREEN_WIDTH = 1020
SCREEN_HEIGHT = 512
SCREEN_TITLE = "Voice-Controlled Arcade with Obstacles"

# Начальная скорость прокрутки
INITIAL_SCROLL_SPEED = 5  # Начальное значение
ACCELERATION_INTERVAL = 5  # Каждые N секунд игра ускоряется
ACCELERATION_AMOUNT = 0.2    # На сколько увеличивается скорость каждые N секунд

# Гравитация
UP_FORCE = 50
GRAVITY_FORCE = 30
MAX_UP_SPEED = 100
MAX_DOWN_SPEED = 100

# Пол и максимальная высота
GROUND_Y = 100
AIR_Y = 300

GAME_DURATION = 60  # 60 секунд

BLOCKSIZE = 1024
PITCH_FLOOR = 100
PITCH_CEILING = 600
SILENCE_THRESHOLD_DB = -25.0
VOICING_THRESHOLD = 0.7
BLOCKS_TO_SILENT = 3

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

# ------------------------- Загрузка пар текстур -------------------------
def load_texture_pair(filename):
    return [
        arcade.load_texture(filename),
        arcade.load_texture(filename, flipped_horizontally=True),
    ]

class ScrollingBackground:
    """
    Бесконечно движущийся фон.
    """
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

    def update(self, scroll_speed):
        for sprite in self.background_list:
            sprite.center_x -= scroll_speed
            if sprite.right < 0:
                sprite.left = self.width

    def draw(self):
        self.background_list.draw()

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

# ------------------------- Препятствие -------------------------
class LavaObstacle(arcade.Sprite):
    """
    Препятствие ("гора с лавой"), движется слева направо, нужно перепрыгивать.
    """
    def __init__(self, speed):
        # Можно нарисовать свою текстуру, здесь используем test.png
        super().__init__("lava1.png", 0.3)

        # Случайная высота
        self.center_x = SCREEN_WIDTH + 50
        # Расположим препятствие снизу, но можно варьировать
        base_height = 64  # минимальная высота (основание)
        self.center_y = base_height // 2  # центр Y

        self.change_x = -speed
        # Можно рандомизировать высоту (чтобы загораживало чуть больше)
        # self.height = random.randint(100, 300)
        # Но arcade.Sprite не так просто меняет height, лучше загружать разные текстуры
        # или масштабировать self.scale.

    def update(self):
        self.center_x += self.change_x
        if self.right < 0:
            self.remove_from_sprite_lists()

# ------------------------- Класс игры -------------------------
class VoiceArcadeGame(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)

        self.scroll_speed = INITIAL_SCROLL_SPEED
        self.background = None
        self.player = None
        self.obstacle_list = None

        self.score = 0
        self.start_time = None
        self.voice_generator = None
        self.game_over = False

        arcade.set_background_color(arcade.csscolor.SKY_BLUE)

    def setup(self):
        self.scroll_speed = INITIAL_SCROLL_SPEED
        self.background = ScrollingBackground(
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            ":resources:images/backgrounds/abstract_1.jpg",
            self.scroll_speed
        )
        self.player = PlayerCharacter()
        self.obstacle_list = arcade.SpriteList()
        self.score = 0
        self.start_time = time.time()
        self.voice_generator = analyze_voice()
        self.game_over = False

    def on_draw(self):
        arcade.start_render()
        self.background.draw()
        self.obstacle_list.draw()
        self.player.draw()

        # Счёт
        arcade.draw_text(f"Score: {self.score}", 10, SCREEN_HEIGHT - 30, arcade.color.BLACK, 20)

        # Таймер
        elapsed = time.time() - self.start_time
        left = GAME_DURATION - elapsed
        arcade.draw_text(f"Time: {int(left)}", SCREEN_WIDTH - 120, SCREEN_HEIGHT - 30, arcade.color.BLACK, 20)

        # GAME OVER
        if self.game_over or left <= 0:
            arcade.draw_text(
                "GAME OVER!", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2,
                arcade.color.RED, 50, anchor_x="center"
            )

    def on_update(self, delta_time: float):
        if self.game_over:
            return

        elapsed = time.time() - self.start_time
        if elapsed > GAME_DURATION:
            self.game_over = True
            return

        # Ускорение каждые ACCELERATION_INTERVAL секунд
        if int(elapsed) % ACCELERATION_INTERVAL == 0:
            # Увеличиваем scroll_speed
            self.scroll_speed += ACCELERATION_AMOUNT

        # Фон двигается
        self.background.update(self.scroll_speed)

        # Препятствия двигаются
        for obs in self.obstacle_list:
            obs.change_x = -self.scroll_speed
        self.obstacle_list.update()

        # Логика голоса
        try:
            voiced_state = next(self.voice_generator)
        except StopIteration:
            voiced_state = 0
        self.player.update_physics(voiced_state == 1)
        self.player.update_animation(delta_time)

        # Счёт
        self.score += 1  # например, +1 за кадр (высокое число) или уместнее +1 в секунду
        # Можно сделать: self.score += int(delta_time * 60)

        # Проверяем столкновения
        if arcade.check_for_collision_with_list(self.player, self.obstacle_list):
            # Столкновение => game over
            self.game_over = True

        # Добавляем препятствия (случайно)
        if random.random() < 0.02:
            obstacle = LavaObstacle(self.scroll_speed)
            # Поменять scale, если нужно.
            self.obstacle_list.append(obstacle)

# ------------------------- Запуск -------------------------
if __name__ == "__main__":
    game = VoiceArcadeGame()
    game.setup()
    arcade.run()
