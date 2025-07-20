import pygame
import sys
import random
import parselmouth
from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32
import numpy as np

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
MAX_JUMP = 50
BLOCKS_TO_SILENT = 2

# Инициализация Pygame
pygame.init()

# Параметры экрана
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
FPS = 60

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (135, 206, 250)

# Параметры шарика
BALL_RADIUS = 15
BALL_TOP_Y = 100  # Верхняя позиция шарика
BALL_BOTTOM_Y = SCREEN_HEIGHT - 100  # Нижняя позиция шарика

# Параметры горы
MOUNTAIN_WIDTH = 50
MOUNTAIN_HEIGHT = 200
MOUNTAIN_SPEED = 3

# Инициализация экрана
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Voice-Controlled Ball Game")
clock = pygame.time.Clock()

# Шрифт для отображения времени
font = pygame.font.SysFont(None, 36)

# Функция для отрисовки шарика
def draw_ball(x, y):
    pygame.draw.circle(screen, BLACK, (x, y), BALL_RADIUS)

# Функция для отрисовки горы
def draw_mountain(mountain):
    mountain_rect = pygame.Rect(mountain['x'], SCREEN_HEIGHT - MOUNTAIN_HEIGHT, MOUNTAIN_WIDTH, MOUNTAIN_HEIGHT)
    pygame.draw.rect(screen, BLACK, mountain_rect)

# Функция для проверки столкновений
def check_collision(ball_x, ball_y, mountain):
    if mountain is None:
        return False
    mountain_rect = pygame.Rect(mountain['x'], SCREEN_HEIGHT - MOUNTAIN_HEIGHT, MOUNTAIN_WIDTH, MOUNTAIN_HEIGHT)
    ball_rect = pygame.Rect(ball_x - BALL_RADIUS, ball_y - BALL_RADIUS, BALL_RADIUS * 2, BALL_RADIUS * 2)
    return mountain_rect.colliderect(ball_rect)

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
        silent_counter = 0
        for _ in range(stream.get_iterations(seconds=30)):  
            raw_data = stream.read(BLOCKSIZE)
            if not raw_data:
                continue
            
            signal = stream.chain_of_methods(raw_data)  
            sound = parselmouth.Sound(values=signal, sampling_frequency=stream.samplerate)
            
            # Анализ интенсивности
            intensity_obj = sound.to_intensity()
            intensity_values = intensity_obj.values.T.flatten()
            avg_intensity = np.mean(intensity_values) if len(intensity_values) > 0 else -50
            
            # Анализ pitch
            if METHOD == "ac":
                pitch_obj = sound.to_pitch_ac(
                    time_step=TIME_STEP, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING,
                    voicing_threshold=VOICING_THRESHOLD, octave_cost=OCTAVE_COST,
                    octave_jump_cost=OCTAVE_JUMP_COST, voiced_unvoiced_cost=VOICED_UNVOICED_COST
                )
            elif METHOD == "cc":
                pitch_obj = sound.to_pitch_cc(
                    time_step=TIME_STEP, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING,
                    voicing_threshold=VOICING_THRESHOLD, octave_cost=OCTAVE_COST,
                    octave_jump_cost=OCTAVE_JUMP_COST, voiced_unvoiced_cost=VOICED_UNVOICED_COST
                )
            else:
                raise ValueError(f"Неизвестный метод: {METHOD}. Используйте 'ac' или 'cc'.")
            
            pitch_values = pitch_obj.selected_array['frequency']
            pitch_values[(pitch_values == 0) | (pitch_values > PITCH_CEILING)] = np.nan
            avg_pitch = np.nanmean(pitch_values) if np.any(~np.isnan(pitch_values)) else np.nan

            # Определение состояния Voiced/Silent
            above_silence_threshold = avg_intensity > SILENCE_THRESHOLD_DB
            valid_pitch = avg_pitch > PITCH_FLOOR
            if above_silence_threshold and valid_pitch:
                current_state = 1
                silent_counter = 0
            else:
                silent_counter += 1
                if silent_counter >= BLOCKS_TO_SILENT:
                    current_state = 0
            yield current_state

# Основной игровой цикл
def main():
    ball_x = 100
    ball_y = BALL_BOTTOM_Y  # Начальная позиция шарика (внизу)

    mountains = []  # Список гор
    score = 0
    game_over = False
    start_time = pygame.time.get_ticks()

    # Запуск анализа голоса
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
            voiced_state = next(voice_generator)
        except StopIteration:
            voiced_state = 0

        # Логика игры
        if not game_over:
            # Мгновенное изменение позиции шарика
            if voiced_state == 1:
                ball_y = BALL_TOP_Y  # Поднимаем шарик наверх
            else:
                ball_y = BALL_BOTTOM_Y  # Опускаем шарик вниз

            # Добавление новой горы
            if random.random() < 0.02:  # Вероятность появления горы
                mountains.append({'x': SCREEN_WIDTH})

            # Движение гор
            for mountain in mountains:
                mountain['x'] -= MOUNTAIN_SPEED

            # Удаление гор, которые вышли за экран
            mountains = [mountain for mountain in mountains if mountain['x'] + MOUNTAIN_WIDTH > 0]

            # Проверка столкновений
            for mountain in mountains:
                if check_collision(ball_x, ball_y, mountain):
                    game_over = True

            # Подсчет очков
            score += 1

        # Отрисовка
        screen.fill(BLUE)  # Фон
        draw_ball(ball_x, ball_y)  # Шарик
        for mountain in mountains:
            draw_mountain(mountain)  # Горы

        # Отображение времени и счета
        time_text = font.render(f"Time: {int(30 - elapsed_time)}", True, WHITE)
        score_text = font.render(f"Score: {score}", True, WHITE)
        screen.blit(time_text, (10, 10))
        screen.blit(score_text, (SCREEN_WIDTH - 120, 10))

        if game_over:
            game_over_text = font.render("Game Over!", True, WHITE)
            screen.blit(game_over_text, (SCREEN_WIDTH // 2 - 80, SCREEN_HEIGHT // 2))

        pygame.display.flip()
        clock.tick(FPS)

# Запуск игры
if __name__ == "__main__":
    main()