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
SCREEN_TITLE = "Voice-Controlled Arcade Game"

# Ускоряем прокрутку фона/объектов
SCROLL_SPEED = 25

# Параметры положения персонажа (минимум и максимум по Y)
GROUND_Y = 100  # координата Y, когда персонаж бежит по земле
AIR_Y = 300     # верхняя точка, куда персонаж может достичь

# ПАРАМЕТРЫ ФИЗИКИ
UP_FORCE = 50      # Сила, с которой персонаж поднимается при слитной речи
GRAVITY_FORCE = 30  # Сила, с которой персонаж опускается при отсутствии речи
MAX_UP_SPEED = 100   # Максимальная скорость подъема
MAX_DOWN_SPEED = 100 # Максимальная скорость падения

WAVE_INTERVAL = 5  # Интервал в секундах между волнами
WAVE_SIZE = 3      # Количество камней в волне


# Параметры длительности игры
GAME_DURATION = 60  # 60 секунд

# Параметры анализа голоса
BLOCKSIZE = 1024
PITCH_FLOOR = 100
PITCH_CEILING = 600
SILENCE_THRESHOLD_DB = 40.0
VOICING_THRESHOLD = 0.6
BLOCKS_TO_SILENT = 4  # если подряд столько блоков тишины, считаем, что голос пропал

# ------------------------- Функция анализа голоса -------------------------
def analyze_voice():
    """
    Генератор, который каждые BLOCKSIZE сэмплов анализирует звук
    и выдает 1 (слитная речь) или 0 (раздельность/тишина).
    """
    with InputStream(samplerate=16000, blocksize=BLOCKSIZE, channels=1, sampwidth=2) as stream:
        stream.set_methods(UnpackRawInFloat32())
        silent_counter = 0
        start_time = time.time()
        while time.time() - start_time < GAME_DURATION:
            raw_data = stream.read(BLOCKSIZE)
            if not raw_data:
                continue
            # Преобразуем в float32 (chain_of_methods)
            signal = stream.chain_of_methods(raw_data)
            # Используем parselmouth
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
                voicing_threshold=VOICING_THRESHOLD,
                octave_cost=0.01,
                octave_jump_cost=0.35,
                voiced_unvoiced_cost=0.14
            )
            pitch_values = pitch_obj.selected_array['frequency']
            pitch_values[(pitch_values == 0) | (pitch_values > PITCH_CEILING)] = np.nan
            avg_pitch = np.nanmean(pitch_values) if np.any(~np.isnan(pitch_values)) else np.nan

            # Логика определения слитности
            above_silence_threshold = avg_intensity > SILENCE_THRESHOLD_DB
            valid_pitch = (avg_pitch > PITCH_FLOOR)
            if above_silence_threshold and valid_pitch:
                silent_counter = 0
                yield 1  # есть голос
            else:
                silent_counter += 1
                if silent_counter >= BLOCKS_TO_SILENT:
                    yield 0  # тишина

# ------------------------- Загрузка пар текстур -------------------------
def load_texture_pair(filename):
    return [
        arcade.load_texture(filename),
        arcade.load_texture(filename, flipped_horizontally=True),
    ]

# ------------------------- Класс фона (ScrollingBackground) -------------------------
class ScrollingBackground:
    """
    Реализует бесконечно движущийся фон.
    """
    def __init__(self, width, height, image_name, scroll_speed):
        self.width = width
        self.height = height
        self.scroll_speed = scroll_speed
        self.background_list = arcade.SpriteList()

        # Два спрайта, которые чередуются
        self.sprite_1 = arcade.Sprite(image_name, scale=1.0)
        self.sprite_2 = arcade.Sprite(image_name, scale=1.0)

        # Положение
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
    """
    Персонаж с анимациями и простой \"мини-физикой\".
    """
    def __init__(self):
        super().__init__()
        # Основной путь к текстурам
        main_path = ":resources:images/animated_characters/male_person/malePerson"

        # Скалирование
        self.scale = 0.5
        self.character_face_direction = 0  # 0 - вправо, 1 - влево
        self.cur_texture = 0

        # Загрузка текстур
        self.idle_texture_pair = load_texture_pair(f"{main_path}_idle.png")
        self.jump_texture_pair = load_texture_pair(f"{main_path}_jump.png")
        self.fall_texture_pair = load_texture_pair(f"{main_path}_fall.png")

        self.walk_textures = []
        for i in range(8):
            texture = load_texture_pair(f"{main_path}_walk{i}.png")
            self.walk_textures.append(texture)

        # Начальная текстура
        self.texture = self.idle_texture_pair[self.character_face_direction]
        self.center_x = 150
        self.center_y = GROUND_Y

        # Хитбокс
        self.hit_box = self.texture.hit_box_points

        # \"мини-физика\" по оси Y
        self.velocity_y = 0.0

    def update_physics(self, is_voiced: bool):
        """
        Меняем скорость в зависимости от голоса и применяем \"гравитацию\".
        """
        if is_voiced:
            # Слитная речь => движение вверх (но ограничиваем сверху)
            self.velocity_y += UP_FORCE
            if self.velocity_y > MAX_UP_SPEED:
                self.velocity_y = MAX_UP_SPEED
        else:
            # Нет речи => опускаемся вниз
            self.velocity_y -= GRAVITY_FORCE
            if self.velocity_y < -MAX_DOWN_SPEED:
                self.velocity_y = -MAX_DOWN_SPEED

        # Применяем скорость
        self.center_y += self.velocity_y

        # Ограничиваем Y
        if self.center_y < GROUND_Y:
            self.center_y = GROUND_Y
            self.velocity_y = 0
        elif self.center_y > AIR_Y:
            self.center_y = AIR_Y
            self.velocity_y = 0

    def update_animation(self, delta_time: float = 1/60):
        """
        Анимация с учетом скорости:
        - velocity_y > 0 => jump
        - velocity_y < 0 => fall
        - velocity_y = 0 => либо walk, если center_y==GROUND, либо idle, если center_y==AIR
        """
        # Если мы на земле
        if self.center_y <= GROUND_Y + 0.1:
            if abs(self.velocity_y) < 0.1:
                # Двигаемся по земле => анимация ходьбы
                self.cur_texture += 1
                if self.cur_texture >= 8:
                    self.cur_texture = 0
                self.texture = self.walk_textures[self.cur_texture][self.character_face_direction]
            else:
                # Переходное состояние (редко) - можем принудительно задать walk
                self.texture = self.walk_textures[self.cur_texture][self.character_face_direction]
        # Если мы вверху
        elif self.center_y >= AIR_Y - 0.1:
            if abs(self.velocity_y) < 0.1:
                # Зависаем вверху => idle
                self.texture = self.idle_texture_pair[self.character_face_direction]
            elif self.velocity_y > 0:
                # Продолжаем подъем (но мы на верхней грани - теоретически 0)
                self.texture = self.jump_texture_pair[self.character_face_direction]
            else:
                # Начинаем падать
                self.texture = self.fall_texture_pair[self.character_face_direction]
        else:
            # Промежуточная высота => смотрим на знак velocity_y
            if self.velocity_y > 0:
                self.texture = self.jump_texture_pair[self.character_face_direction]
            else:
                self.texture = self.fall_texture_pair[self.character_face_direction]

# ------------------------- Класс \"камня\" (Gem) -------------------------
class Gem(arcade.Sprite):
    """
    Заменяем кольца на «камни» (gem). Двигаются влево, как и кольца.
    """
    def __init__(self):
        super().__init__("ringsonik.png", scale=0.1)
        self.center_x = SCREEN_WIDTH + random.randint(50, 300)
        self.center_y = AIR_Y  # камень на уровне прыжка
        self.change_x = -SCROLL_SPEED

    def update(self):
        self.center_x += self.change_x
        # Если камень ушел слишком влево, удаляем
        if self.right < 0:
            self.remove_from_sprite_lists()

# ------------------------- Класс игры (VoiceArcadeGame) -------------------------
class VoiceArcadeGame(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)

        # Фон
        self.background = None
        # Персонаж
        self.player = None
        # Список \"камней\"
        self.gem_list = None

        # Механика \"волны\": нужно собрать ВСЕ камни
        self.wave_in_progress = False     # Флаг, идет ли сейчас волна
        self.wave_size = 0               # Сколько камней в волне
        self.wave_collected = 0          # Сколько камней собрано
        self.wave_failed = False         # Пропущен ли хоть один камень

        # Очки
        self.score = 0
        # Таймер запуска игры
        self.start_time = None
        # Генератор голосового анализа
        self.voice_generator = None

        # Цвет фона
        arcade.set_background_color(arcade.csscolor.SKY_BLUE)

    def setup(self):
        """Начальная настройка игры"""
        # Создаем фон
        self.background = ScrollingBackground(
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
            ":resources:images/backgrounds/abstract_1.jpg",
            SCROLL_SPEED
        )
        # Создаем игрока
        self.player = PlayerCharacter()
        # Список камней
        self.gem_list = arcade.SpriteList()

        # Сброс механики волн
        self.wave_in_progress = False
        self.wave_size = 0
        self.wave_collected = 0
        self.wave_failed = False

        # Сброс очков
        self.score = 0
        # Время старта
        self.start_time = time.time()
        # Генератор голосового анализа
        self.voice_generator = analyze_voice()

    def on_draw(self):
        arcade.start_render()
        # Рисуем фон
        self.background.draw()
        # Рисуем камни
        self.gem_list.draw()
        # Рисуем игрока
        self.player.draw()

        # Рисуем очки
        arcade.draw_text(f"Score: {self.score}", 10, SCREEN_HEIGHT - 30, arcade.color.BLACK, 20)
        # Таймер
        elapsed = time.time() - self.start_time
        left = GAME_DURATION - elapsed
        arcade.draw_text(f"Time: {int(left)}", SCREEN_WIDTH - 120, SCREEN_HEIGHT - 30, arcade.color.BLACK, 20)

        # Если время вышло
        if left <= 0:
            arcade.draw_text(
                "GAME OVER!", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2,
                arcade.color.RED, 40, anchor_x="center"
            )

    def on_update(self, delta_time):
        # Если время вышло, завершаем обновление
        elapsed = time.time() - self.start_time
        if elapsed > GAME_DURATION:
            return  # перестаем обновлять игру

        # Фон двигается
        self.background.update()
        # Камни двигаются
        self.gem_list.update()

        # Получаем состояние голоса
        try:
            voiced_state = next(self.voice_generator)
        except StopIteration:
            voiced_state = 0

        # Обновляем \"мини-физику\" персонажа
        is_voiced = (voiced_state == 1)
        self.player.update_physics(is_voiced)

        # Обновляем анимацию персонажа
        self.player.update_animation(delta_time)

        # Проверка столкновений
        for gem in self.gem_list:
            # Если этот камень ушел за экран - считаем, что волна провалена
            if gem.right < 0:
                self.wave_failed = True
            # Если столкнулись с игроком
            if arcade.check_for_collision(self.player, gem):
                gem.remove_from_sprite_lists()
                self.wave_collected += 1

        # Если волна активна и все камни пропали (или собраны)
        if self.wave_in_progress and len(self.gem_list) == 0:
            # Проверяем, собрали ли все
            if not self.wave_failed and self.wave_collected == self.wave_size:
                self.score += 10  # Начисляем очки только при условии, что собрали все
            # Завершаем волну
            self.wave_in_progress = False

        # Если волна не активна - создаем новую
        if not self.wave_in_progress:
            self.start_new_wave()

    def start_new_wave(self):
        """
        Создаем \"волну\" из нескольких камней сразу,
        чтобы игрок получил очки только при сборе всех.
        """
        self.wave_in_progress = True
        self.wave_failed = False
        self.wave_collected = 0
        # Генерируем случайное кол-во камней, например 2..4
        self.wave_size = random.randint(2, 4)

        for _ in range(self.wave_size):
            new_gem = Gem()
            self.gem_list.append(new_gem)

# ------------------------- Запуск -------------------------
if __name__ == "__main__":
    game = VoiceArcadeGame()
    game.setup()
    arcade.run()
