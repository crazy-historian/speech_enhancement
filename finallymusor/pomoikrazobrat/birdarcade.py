import arcade
import numpy as np
import parselmouth
from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32
import time  # Используем модуль time для отслеживания времени

# Параметры обработки голоса
METHOD = "ac"
PITCH_FLOOR = 75
PITCH_CEILING = 600
TIME_STEP = 0.01
SILENCE_THRESHOLD = 0.03
VOICING_THRESHOLD = 0.45
OCTAVE_COST = 0.01
OCTAVE_JUMP_COST = 0.35
VOICED_UNVOICED_COST = 0.14
BLOCKSIZE = 1024

# Параметры экрана
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
FPS = 60

# Цвета
WHITE = arcade.color.WHITE
BLACK = arcade.color.BLACK
BLUE = arcade.color.SKY_BLUE
GREEN = arcade.color.GREEN
RED = arcade.color.RED

# Параметры птички
BALL_RADIUS = 15
BALL_GRAVITY = 9.8  # Гравитация
BALL_Y_MIN = 100
BALL_Y_MAX = SCREEN_HEIGHT - 100

# Диапазон частот
FREQ_MIN = 100
FREQ_MAX = 200

# Преобразование частоты в координату Y на экране
def map_frequency_to_screen(frequency):
    return BALL_Y_MAX - ((frequency - FREQ_MIN) / (FREQ_MAX - FREQ_MIN)) * (BALL_Y_MAX - BALL_Y_MIN)

class Game(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, "Voice-Controlled Ball Game")
        self.ball_x = 100
        self.ball_y = BALL_Y_MAX
        self.score = 0
        self.start_time = None
        self.voice_generator = None

    def setup(self):
        # Запуск анализа голоса
        self.voice_generator = analyze_voice()
        self.start_time = time.time()  # Используем time.time() для отслеживания времени

    def on_draw(self):
        self.clear()
        arcade.set_background_color(BLUE)

        # Отрисовка диапазона частот
        freq_min_y = map_frequency_to_screen(FREQ_MIN)
        freq_max_y = map_frequency_to_screen(FREQ_MAX)

        # Поменяем местами freq_min_y и freq_max_y, чтобы bottom <= top
        bottom_y = min(freq_min_y, freq_max_y)
        top_y = max(freq_min_y, freq_max_y)

        arcade.draw_lrbt_rectangle_filled(0, SCREEN_WIDTH, bottom_y, top_y, GREEN)

        # Отрисовка птички
        arcade.draw_circle_filled(self.ball_x, self.ball_y, BALL_RADIUS, BLACK)

        # Отображение времени и счета
        elapsed_time = time.time() - self.start_time
        time_text = f"Time: {int(30 - elapsed_time)}"
        score_text = f"Score: {self.score}"
        arcade.draw_text(time_text, 10, SCREEN_HEIGHT - 30, WHITE, 16)
        arcade.draw_text(score_text, SCREEN_WIDTH - 120, SCREEN_HEIGHT - 30, WHITE, 16)

        # Отображение границ диапазона
        freq_min_text = f"{FREQ_MIN} Hz"
        freq_max_text = f"{FREQ_MAX} Hz"
        arcade.draw_text(freq_min_text, 10, freq_min_y - 20, WHITE, 12)
        arcade.draw_text(freq_max_text, 10, freq_max_y + 10, WHITE, 12)

    def on_update(self, delta_time):
        elapsed_time = time.time() - self.start_time

        # Завершение игры через 30 секунд
        if elapsed_time >= 30:
            print(f"Игра завершена! Ваш счет: {self.score}")
            arcade.close_window()
            return

        # Анализ голоса
        try:
            pitch = next(self.voice_generator)
        except StopIteration:
            pitch = np.nan

        # Логика игры
        if np.isnan(pitch):
            # Если сигнала нет, птичка падает под действием гравитации
            self.ball_y = min(self.ball_y + BALL_GRAVITY * delta_time, BALL_Y_MAX)
        else:
            # Преобразуем частоту в координату Y
            target_y = map_frequency_to_screen(pitch)
            target_y = max(BALL_Y_MIN, min(target_y, BALL_Y_MAX))

            # Перемещаем птичку к целевой позиции
            self.ball_y = arcade.math.lerp(self.ball_y, target_y, 0.1)

            # Проверяем, находится ли частота в нужном диапазоне
            if FREQ_MIN <= pitch <= FREQ_MAX:
                self.score += 1

    def close(self):
        super().close()

# Анализ голоса
def analyze_voice():
    with InputStream(
        samplerate=16000,
        blocksize=BLOCKSIZE,
        channels=1,
        sampwidth=2
    ) as stream:
        print(f"Захват звука с микрофона. Частота дискретизации: {stream.samplerate} Гц")
        stream.set_methods(UnpackRawInFloat32())
        for _ in range(stream.get_iterations(seconds=30)):
            raw_data = stream.read(BLOCKSIZE)
            if not raw_data:
                yield np.nan
                continue

            signal = stream.chain_of_methods(raw_data)
            sound = parselmouth.Sound(values=signal, sampling_frequency=stream.samplerate)

            if METHOD == "ac":
                pitch_obj = sound.to_pitch_ac(
                    time_step=TIME_STEP, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING,
                    silence_threshold=SILENCE_THRESHOLD, voicing_threshold=VOICING_THRESHOLD,
                    octave_cost=OCTAVE_COST, octave_jump_cost=OCTAVE_JUMP_COST,
                    voiced_unvoiced_cost=VOICED_UNVOICED_COST
                )
            elif METHOD == "cc":
                pitch_obj = sound.to_pitch_cc(
                    time_step=TIME_STEP, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING,
                    silence_threshold=SILENCE_THRESHOLD, voicing_threshold=VOICING_THRESHOLD,
                    octave_cost=OCTAVE_COST, octave_jump_cost=OCTAVE_JUMP_COST,
                    voiced_unvoiced_cost=VOICED_UNVOICED_COST
                )
            else:
                raise ValueError(f"Неизвестный метод: {METHOD}. Используйте 'ac' или 'cc'.")

            pitch_values = pitch_obj.selected_array['frequency']
            pitch_values[(pitch_values == 0) | (pitch_values > PITCH_CEILING)] = np.nan
            avg_pitch = np.nanmean(pitch_values) if np.any(~np.isnan(pitch_values)) else np.nan
            yield avg_pitch

# Запуск игры
if __name__ == "__main__":
    game = Game()
    game.setup()
    arcade.run()