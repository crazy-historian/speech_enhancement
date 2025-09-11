import pygame
import sys
import numpy as np
import parselmouth
from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32

# Параметры обработки
METHOD = "cc"
PITCH_FLOOR = 100
PITCH_CEILING = 600
TIME_STEP = 0.01
SILENCE_THRESHOLD_DB = -25.0
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
BALL_GRAVITY = 0.2  # Умеренная скорость падения
BALL_Y_MIN = 100    # Минимальная высота птички
BALL_Y_MAX = SCREEN_HEIGHT - 100  # Максимальная высота птички

# Диапазон интенсивности
INTENSITY_MIN = 20  # Нижняя граница диапазона (в дБ)
INTENSITY_MAX = 150  # Верхняя граница диапазона (в дБ)

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

# Функция для отрисовки диапазона интенсивности
def draw_intensity_range():
    intensity_min_y = map_intensity_to_screen(INTENSITY_MIN)
    intensity_max_y = map_intensity_to_screen(INTENSITY_MAX)
    pygame.draw.rect(screen, GREEN, (0, intensity_min_y, SCREEN_WIDTH, intensity_max_y - intensity_min_y))

# Функция для преобразования интенсивности в координату Y на экране
def map_intensity_to_screen(intensity):
    # Масштабируем интенсивность в диапазон [BALL_Y_MIN, BALL_Y_MAX]
    return BALL_Y_MAX - ((intensity - INTENSITY_MIN) / (INTENSITY_MAX - INTENSITY_MIN)) * (BALL_Y_MAX - BALL_Y_MIN)

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
            
            # Анализ интенсивности
            intensity_obj = sound.to_intensity()
            intensity_values = intensity_obj.values.T.flatten()
            avg_intensity = np.mean(intensity_values) if len(intensity_values) > 0 else np.nan
            yield avg_intensity

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
            intensity = next(voice_generator)
        except StopIteration:
            intensity = np.nan

        # Логика игры
        if np.isnan(intensity):
            # Если сигнала нет, птичка плавно падает
            ball_y = min(ball_y + BALL_GRAVITY, BALL_Y_MAX)
        else:
            # Преобразуем интенсивность в координату Y
            ball_y = map_intensity_to_screen(intensity)
            ball_y = max(BALL_Y_MIN, min(ball_y, BALL_Y_MAX))  # Ограничиваем высоту птички

            # Проверяем, находится ли интенсивность в нужном диапазоне
            if INTENSITY_MIN <= intensity <= INTENSITY_MAX:
                score += 1

        # Отрисовка
        screen.fill(BLUE)  # Фон
        draw_intensity_range()  # Выделение диапазона интенсивности
        draw_ball(ball_x, ball_y)  # Птичка

        # Отображение времени и счета
        time_text = font.render(f"Time: {int(30 - elapsed_time)}", True, WHITE)
        score_text = font.render(f"Score: {score}", True, WHITE)
        screen.blit(time_text, (10, 10))
        screen.blit(score_text, (SCREEN_WIDTH - 120, 10))

        # Отображение границ диапазона
        intensity_min_text = font.render(f"{INTENSITY_MIN} dB", True, WHITE)
        intensity_max_text = font.render(f"{INTENSITY_MAX} dB", True, WHITE)
        screen.blit(intensity_min_text, (10, map_intensity_to_screen(INTENSITY_MIN) - 20))
        screen.blit(intensity_max_text, (10, map_intensity_to_screen(INTENSITY_MAX) + 10))

        pygame.display.flip()
        clock.tick(FPS)

# Запуск игры
if __name__ == "__main__":
    main()