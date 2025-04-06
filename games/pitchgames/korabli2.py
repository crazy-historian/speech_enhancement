import arcade
import time
import parselmouth
import numpy as np
from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32

# Настройки экрана
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 512
SCREEN_TITLE = "Voice Ship Control"

# Параметры движения
START_X = 100
END_X = 1000
SHIP_SPEED = (END_X - START_X) / 2

# Зона pitch
PITCH_MIN = 110
PITCH_MAX = 160
SMOOTHING_FACTOR = 0.2  # <--- сглаживание движения по Y

# Аудио
BLOCKSIZE = 1024
SILENCE_THRESHOLD_DB = 60.0
BLOCKS_TO_SILENT = 4
PITCH_FLOOR = 90
PITCH_CEILING = 300
VOICING_THRESHOLD = 0.6

# Максимальное пребывание за пределами
MAX_OUT_OF_ZONE_DURATION = 1.0


class Ship(arcade.SpriteSolidColor):
    def __init__(self):
        super().__init__(50, 30, arcade.color.BLUE)
        self.center_x = START_X
        self.center_y = SCREEN_HEIGHT // 2
        self.moving_forward = False
        self.out_of_zone = False

    def update_position(self, voice_status, pitch, delta_time):
        if voice_status == 1 and PITCH_MIN <= pitch <= PITCH_MAX:
            self.moving_forward = True
            pitch_range = PITCH_MAX - PITCH_MIN
            relative_position = (pitch - PITCH_MIN) / pitch_range
            target_y = SCREEN_HEIGHT // 2 - 80 + relative_position * 160
            self.center_y += (target_y - self.center_y) * SMOOTHING_FACTOR
        elif voice_status == 1:
            self.moving_forward = True  # движется даже если вышел из зоны
            direction = "up" if pitch > PITCH_MAX else "down"
            overshoot = abs(pitch - (PITCH_MAX if direction == "up" else PITCH_MIN)) / 100
            offset = overshoot * (SCREEN_HEIGHT // 2 - 100)
            target_y = SCREEN_HEIGHT // 2 + 80 + offset if direction == "up" else SCREEN_HEIGHT // 2 - 80 - offset
            self.center_y += (target_y - self.center_y) * SMOOTHING_FACTOR
        else:
            self.moving_forward = False
            self.center_y += (SCREEN_HEIGHT // 2 - self.center_y) * 0.1

        if self.moving_forward and self.center_x < END_X:
            self.center_x += SHIP_SPEED * delta_time


class VoiceShipGame(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        arcade.set_background_color(arcade.color.SKY_BLUE)
        self.ship = Ship()
        self.voice_generator = analyze_voice()
        self.current_pitch = None
        self.started = False
        self.game_over = False
        self.success = False
        self.out_of_zone_timer = 0.0
        self.timer_start_time = None
        self.elapsed_time = 0

    def reset_game(self):
        self.ship = Ship()
        self.voice_generator = analyze_voice()
        self.current_pitch = None
        self.started = False
        self.game_over = False
        self.success = False
        self.out_of_zone_timer = 0.0

    def on_draw(self):
        arcade.start_render()
        self.ship.draw()

        # Зоны
        arcade.draw_rectangle_outline(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, SCREEN_WIDTH, 160, arcade.color.GREEN, 2)
        arcade.draw_rectangle_outline(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 80, SCREEN_WIDTH, 160, arcade.color.RED, 2)
        arcade.draw_rectangle_outline(SCREEN_WIDTH // 2, 80, SCREEN_WIDTH, 160, arcade.color.RED, 2)

        # Победа
        if self.success:
            arcade.draw_text("SUCCESS!", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, arcade.color.GREEN, 50, anchor_x="center")

        # Проигрыш
        if self.game_over:
            arcade.draw_text("GAME OVER", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, arcade.color.RED, 50, anchor_x="center")
        
        if self.timer_start_time:
            arcade.draw_text(f"Time: {self.elapsed_time:.2f}s", SCREEN_WIDTH // 2, SCREEN_HEIGHT - 60, arcade.color.BLACK, 20, anchor_x="center")

        # Шкала pitch
        self.draw_pitch_scale()

    def draw_pitch_scale(self):
        scale_x = SCREEN_WIDTH - 50
        top = SCREEN_HEIGHT - 20
        bottom = 20
        height = top - bottom

        arcade.draw_line(scale_x, bottom, scale_x, top, arcade.color.BLACK, 2)

        for i in range(6):
            freq = PITCH_FLOOR + i * (PITCH_CEILING - PITCH_FLOOR) / 5
            y = bottom + (i / 5) * height
            arcade.draw_line(scale_x - 10, y, scale_x + 10, y, arcade.color.BLACK)
            arcade.draw_text(f"{int(freq)}", scale_x + 15, y - 8, arcade.color.BLACK, 12)

        y_min = bottom + ((PITCH_MIN - PITCH_FLOOR) / (PITCH_CEILING - PITCH_FLOOR)) * height
        y_max = bottom + ((PITCH_MAX - PITCH_FLOOR) / (PITCH_CEILING - PITCH_FLOOR)) * height
        arcade.draw_rectangle_filled(scale_x, (y_min + y_max) / 2, 20, y_max - y_min, arcade.color.ALMOND)

        if self.current_pitch:
            y_current = bottom + ((self.current_pitch - PITCH_FLOOR) / (PITCH_CEILING - PITCH_FLOOR)) * height
            if bottom <= y_current <= top:
                arcade.draw_line(scale_x - 15, y_current, scale_x + 15, y_current, arcade.color.RED, 3)

    def on_update(self, delta_time: float):
        if self.game_over or self.success:
            return

        try:
            voice_status, pitch = next(self.voice_generator)
        except StopIteration:
            return

        if not self.started and voice_status == 1:
            self.started = True

        if not self.started:
            return
        
        if self.timer_start_time is None and voice_status == 1:
            self.timer_start_time = time.time()

        if self.timer_start_time and not self.game_over and self.ship.center_x < END_X:
            self.elapsed_time = time.time() - self.timer_start_time

        self.current_pitch = pitch
        self.ship.update_position(voice_status, pitch, delta_time)

        # Проверка на победу
        if self.ship.center_x >= END_X and PITCH_MIN <= pitch <= PITCH_MAX:
            self.success = True
            return

        # Проверка на проигрыш
        if voice_status == 0:
            self.game_over = True
            return

        if pitch < PITCH_MIN or pitch > PITCH_MAX:
            self.out_of_zone_timer += delta_time
            if self.out_of_zone_timer >= MAX_OUT_OF_ZONE_DURATION:
                self.game_over = True
        else:
            self.out_of_zone_timer = 0.0

    def on_key_press(self, key, modifiers):
        if key == arcade.key.R:
            self.reset_game()
            self.timer_start_time = None
            self.elapsed_time = 0
            



def analyze_voice():
    with InputStream(samplerate=16000, blocksize=BLOCKSIZE, channels=1, sampwidth=2) as stream:
        stream.set_methods(UnpackRawInFloat32())
        last_valid_pitch = None
        silent_counter = 0
        while True:
            raw_data = stream.read(BLOCKSIZE)
            if not raw_data:
                continue
            signal = stream.chain_of_methods(raw_data)
            sound = parselmouth.Sound(values=signal, sampling_frequency=stream.samplerate)

            intensity = np.mean(sound.to_intensity().values) if sound.to_intensity().values.size > 0 else -50
            pitch_values = sound.to_pitch_ac(pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING,
                                             voicing_threshold=VOICING_THRESHOLD).selected_array['frequency']
            pitch_values[(pitch_values == 0) | (pitch_values > PITCH_CEILING)] = np.nan
            pitch = np.nanmean(pitch_values) if np.any(~np.isnan(pitch_values)) else last_valid_pitch

            if intensity > SILENCE_THRESHOLD_DB and pitch is not None:
                last_valid_pitch = pitch
                silent_counter = 0
                yield 1, pitch
            else:
                silent_counter += 1
                if last_valid_pitch is not None:
                    yield 1, last_valid_pitch
                else:
                    yield 0, None  # если pitch ещё не получен — блокируем старт

                if silent_counter >= BLOCKS_TO_SILENT:
                    yield 0, None


if __name__ == "__main__":
    game = VoiceShipGame()
    arcade.run()
