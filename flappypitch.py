import pygame
import sys
import numpy as np
import parselmouth
from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32

# Параметры обработки
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
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (135, 206, 250)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Параметры птички
BALL_RADIUS = 15
BALL_GRAVITY = 10  # Умеренная скорость падения
BALL_Y_MIN = 100    # Минимальная высота птички
BALL_Y_MAX = SCREEN_HEIGHT - 100  # Максимальная высота птички

# Диапазон частот
FREQ_MIN = 100  # Нижняя граница диапазона
FREQ_MAX = 200  # Верхняя граница диапазона

# Инициализация Pygame
pygame.init()

# Инициализация экрана
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Voice-Controlled Ball Game")
clock = pygame.time.Clock()

# Шрифт для отображения времени, счета и диапазона
font = pygame.font.SysFont(None, 36)

# Функция для отрисовки птички
def draw_ball(x, y):
    pygame.draw.circle(screen, BLACK, (x, y), BALL_RADIUS)

# Функция для отрисовки диапазона частот
def draw_frequency_range():
    freq_min_y = map_frequency_to_screen(FREQ_MIN)
    freq_max_y = map_frequency_to_screen(FREQ_MAX)
    pygame.draw.rect(screen, GREEN, (0, freq_min_y, SCREEN_WIDTH, freq_max_y - freq_min_y))

# Функция для преобразования частоты в координату Y на экране
def map_frequency_to_screen(frequency):
    # Масштабируем частоту в диапазон [BALL_Y_MIN, BALL_Y_MAX]
    return BALL_Y_MAX - ((frequency - FREQ_MIN) / (FREQ_MAX - FREQ_MIN)) * (BALL_Y_MAX - BALL_Y_MIN)

# Захват звука и анализ голоса
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
                yield np.nan  # Если данных нет, возвращаем NaN
                continue
            
            signal = stream.chain_of_methods(raw_data)  
            sound = parselmouth.Sound(values=signal, sampling_frequency=stream.samplerate)
            
            # Анализ pitch
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

# Основной игровой цикл
def main():
    ball_x = 100
    ball_y = BALL_Y_MAX  # Начальная позиция птички (внизу)

    score = 0
    start_time = pygame.time.get_ticks()
    voice_generator = analyze_voice()

    while True:
        current_time = pygame.time.get_ticks()
        elapsed_time = (current_time - start_time) / 1000  # Время в секундах

        # Завершение игры через 30 секунд
        if elapsed_time >= 30:
            print(f"Игра завершена! Ваш счет: {score}")
            break

        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Анализ голоса
        try:
            pitch = next(voice_generator)
        except StopIteration:
            pitch = np.nan

        # Логика игры
        if np.isnan(pitch):
            # Если сигнала нет, птичка плавно падает
            ball_y = min(ball_y + BALL_GRAVITY, BALL_Y_MAX)
        else:
            # Преобразуем частоту в координату Y
            ball_y = map_frequency_to_screen(pitch)
            ball_y = max(BALL_Y_MIN, min(ball_y, BALL_Y_MAX))  # Ограничиваем высоту птички

            # Проверяем, находится ли частота в нужном диапазоне
            if FREQ_MIN <= pitch <= FREQ_MAX:
                score += 1

        # Отрисовка
        screen.fill(BLUE)  # Фон
        draw_frequency_range()  # Выделение диапазона частот
        draw_ball(ball_x, ball_y)  # Птичка

        # Отображение времени и счета
        time_text = font.render(f"Time: {int(30 - elapsed_time)}", True, WHITE)
        score_text = font.render(f"Score: {score}", True, WHITE)
        screen.blit(time_text, (10, 10))
        screen.blit(score_text, (SCREEN_WIDTH - 120, 10))

        # Отображение границ диапазона
        freq_min_text = font.render(f"{FREQ_MIN} Hz", True, WHITE)
        freq_max_text = font.render(f"{FREQ_MAX} Hz", True, WHITE)
        screen.blit(freq_min_text, (10, map_frequency_to_screen(FREQ_MIN) - 20))
        screen.blit(freq_max_text, (10, map_frequency_to_screen(FREQ_MAX) + 10))

        pygame.display.flip()
        clock.tick(FPS)

# Запуск игры
if __name__ == "__main__":
    main()