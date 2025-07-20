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
#больше плавности
#сделать прототип интерфейса
# Настройки корабля
START_X = 100
END_X = 1000
SHIP_SPEED = (END_X - START_X) / 4  # пикселей в секунду (2 секунды до цели)

# Частотная зона (в Hz)
PITCH_MIN = 110
PITCH_MAX = 160

# Аудио
BLOCKSIZE = 1024
SILENCE_THRESHOLD_DB = 60.0
BLOCKS_TO_SILENT = 1
PITCH_FLOOR = 90
PITCH_CEILING = 300
VOICING_THRESHOLD = 0.6

# Время игры
GAME_DURATION = 30

class Ship(arcade.SpriteSolidColor):
    def __init__(self):
        super().__init__(50, 30, arcade.color.BLUE)
        self.center_x = START_X
        self.center_y = SCREEN_HEIGHT // 2
        self.moving_forward = False
        self.out_of_zone = False
        self.out_direction = None
        

    def update_position(self, voice_status, pitch, delta_time):
        # В зелёной зоне
        if voice_status == 1 and PITCH_MIN <= pitch <= PITCH_MAX:
            self.moving_forward = True
            self.out_of_zone = False
            self.out_direction = None

            pitch_range = PITCH_MAX - PITCH_MIN
            if pitch_range > 0:
                relative_position = (pitch - PITCH_MIN) / pitch_range
                zone_center_y = SCREEN_HEIGHT // 2
                zone_half_height = 80
                target_y = zone_center_y - zone_half_height + (relative_position * 160)
                self.center_y += (target_y - self.center_y) * 0.2

        # Вышел вверх
        elif voice_status == 1 and pitch > PITCH_MAX:
            self.moving_forward = False
            self.out_of_zone = True
            self.out_direction = "up"

            # Плавно двигаем по диагонали назад и вверх
            if self.center_x > START_X:
                self.center_x -= SHIP_SPEED * delta_time
            overshoot = min((pitch - PITCH_MAX) / 100, 1.0)
            target_y = SCREEN_HEIGHT // 2 + 80 + overshoot * (SCREEN_HEIGHT // 2 - 100)
            self.center_y += (target_y - self.center_y) * 0.1

        # Вышел вниз
        elif voice_status == 1 and pitch < PITCH_MIN:
            self.moving_forward = False
            self.out_of_zone = True
            self.out_direction = "down"

            # Плавно двигаем по диагонали назад и вниз
            if self.center_x > START_X:
                self.center_x -= SHIP_SPEED * delta_time
            overshoot = min((PITCH_MIN - pitch) / 100, 1.0)
            target_y = SCREEN_HEIGHT // 2 - 80 - overshoot * (SCREEN_HEIGHT // 2 - 100)
            self.center_y += (target_y - self.center_y) * 0.1

        else:
            # Тишина
            self.moving_forward = False
            self.out_of_zone = False
            self.out_direction = None

            # Плавное возвращение в центр, если нет сигнала
            if self.center_x > START_X:
                self.center_x -= 1000 * delta_time
                self.center_x = max(self.center_x, START_X)
            self.center_y += (SCREEN_HEIGHT // 2 - self.center_y) * 0.1

        # Движение вперёд разрешено только в зоне
        if self.moving_forward and self.center_x < END_X:
            self.center_x += SHIP_SPEED * delta_time

class VoiceShipGame(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        arcade.set_background_color(arcade.color.SKY_BLUE)
        self.ship = Ship()
        self.start_time = time.time()
        self.voice_generator = analyze_voice()
        self.movement_start_time = None
        self.current_timer = 0
        self.current_pitch = None

    def on_draw(self):
        arcade.start_render()
        self.ship.draw()

        # Расширенная зона (160 пикселей)
        arcade.draw_rectangle_outline(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, SCREEN_WIDTH, 160, arcade.color.GREEN, 2)
        arcade.draw_rectangle_outline(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 80, SCREEN_WIDTH, 160, arcade.color.RED, 2)
        arcade.draw_rectangle_outline(SCREEN_WIDTH // 2, 80, SCREEN_WIDTH, 160, arcade.color.RED, 2)

        # Таймер движения
        arcade.draw_text(f"Moving time: {self.current_timer:.2f}s", SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30,
                         arcade.color.BLACK, 24, anchor_x="center")

        # Частотная шкала
        self.draw_pitch_scale()

        # Победа
        if self.ship.center_x >= END_X:
            arcade.draw_text("Success!", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2,
                             arcade.color.GREEN, 40, anchor_x="center")

    def draw_pitch_scale(self):
        scale_x = SCREEN_WIDTH - 50
        scale_top = SCREEN_HEIGHT - 20
        scale_bottom = 20
        scale_height = scale_top - scale_bottom

        # Шкала
        arcade.draw_line(scale_x, scale_bottom, scale_x, scale_top, arcade.color.BLACK, 2)

        # Деления
        for i in range(6):
            freq = PITCH_FLOOR + i * (PITCH_CEILING - PITCH_FLOOR) / 5
            y = scale_bottom + (i / 5) * scale_height
            arcade.draw_line(scale_x - 10, y, scale_x + 10, y, arcade.color.BLACK, 1)
            arcade.draw_text(f"{int(freq)}", scale_x + 15, y - 10, arcade.color.BLACK, 12)

        # Целевая зона
        y_min = scale_bottom + ((PITCH_MIN - PITCH_FLOOR) / (PITCH_CEILING - PITCH_FLOOR)) * scale_height
        y_max = scale_bottom + ((PITCH_MAX - PITCH_FLOOR) / (PITCH_CEILING - PITCH_FLOOR)) * scale_height
        arcade.draw_rectangle_filled(scale_x, (y_min + y_max) / 2, 20, y_max - y_min, arcade.color.ALMOND)

        # Текущая частота
        if self.current_pitch:
            y_current = scale_bottom + ((self.current_pitch - PITCH_FLOOR) / (PITCH_CEILING - PITCH_FLOOR)) * scale_height
            if scale_bottom <= y_current <= scale_top:
                arcade.draw_line(scale_x - 15, y_current, scale_x + 15, y_current, arcade.color.RED, 3)

    def on_update(self, delta_time: float):
        if time.time() - self.start_time >= GAME_DURATION:
            return
        try:
            voice_status, pitch = next(self.voice_generator)
        except StopIteration:
            voice_status, pitch = 0, None

        self.current_pitch = pitch

        if voice_status == 1 and PITCH_MIN <= pitch <= PITCH_MAX:
            if self.movement_start_time is None:
                self.movement_start_time = time.time()
            self.current_timer = time.time() - self.movement_start_time
        else:
            self.movement_start_time = None
            self.current_timer = 0

        self.ship.update_position(voice_status, pitch, delta_time)

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
                if silent_counter >= BLOCKS_TO_SILENT:
                    yield 0, None

if __name__ == "__main__":
    game = VoiceShipGame()
    arcade.run()
