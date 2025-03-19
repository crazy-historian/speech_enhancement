import arcade
import random
import time
import numpy as np
import parselmouth
from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QSpinBox, QPushButton, QLineEdit
import sys

# ------------------------- Глобальные настройки -------------------------
SCREEN_WIDTH = 1020
SCREEN_HEIGHT = 512
SCREEN_TITLE = "Voice-Controlled Arcade Game"

# Ускоряем прокрутку фона/объектов
SCROLL_SPEED = 25
COLLECTION_TIME=3
# Параметры положения персонажа (минимум и максимум по Y)
GROUND_Y = 100  # координата Y, когда персонаж бежит по земле
AIR_Y = 300     # верхняя точка, куда персонаж может достичь

# ПАРАМЕТРЫ ФИЗИКИ
UP_FORCE = 50      # Сила, с которой персонаж поднимается при слитной речи
GRAVITY_FORCE = 30  # Сила, с которой персонаж опускается при отсутствии речи
MAX_UP_SPEED = 100   # Максимальная скорость подъема
MAX_DOWN_SPEED = 100 # Максимальная скорость падения

WAVE_INTERVAL = 3  # Интервал в секундах между волнами
WAVE_SIZE = 3      # Количество камней в волне


# Параметры длительности игры
GAME_DURATION = 60  # 60 секунд

# Параметры анализа голоса
BLOCKSIZE = 1024
PITCH_FLOOR = 100
PITCH_CEILING = 600
SILENCE_THRESHOLD_DB = 40.0
VOICING_THRESHOLD = 0.6
BLOCKS_TO_SILENT = 2  # если подряд столько блоков тишины, считаем, что голос пропал

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
        """Сохраняет параметры и запускает игру"""
        global WAVE_INTERVAL, WAVE_SIZE, GAME_DURATION, COLLECTION_TIME

        WAVE_INTERVAL = int(self.inputs["wave_interval"].text())
        WAVE_SIZE = int(self.inputs["wave_size"].text())
        GAME_DURATION = int(self.inputs["game_duration"].text())
        COLLECTION_TIME = int(self.inputs["collection_time"].text())
        

        self.close()  # Закрываем окно

        
# ---------------------- Запуск окна перед игрой ----------------------
app = QApplication(sys.argv)
config_window = GameConfigWindow()
config_window.show()
app.exec()

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
    def __init__(self, x, y):
        super().__init__("ringsonik.png", scale=0.1)
        self.center_x = x
        self.center_y = y
        self.change_x = -SCROLL_SPEED
        self.game=game
        

    def update(self):
        self.center_x += self.change_x
        # Если камень ушел слишком влево, удаляем
        
        # if self.right < 0:
        #     self.remove_from_sprite_lists()
        #     self.game.wave_failed=True
        
        # if self.game.wave_in_progress and (self.game.wave_collected == self.game.wave_size or self.game.wave_failed):
        #     print(f"Количество камней перед проверкой: ПЛЮС БАЛЛ")
        #     if not self.game.wave_failed:
                
                
        #         self.game.score += 10  # Очки начисляются только если все камни собраны
        #     self.game.wave_in_progress = False  # Завершаем волну

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
        self.gem_count=0
        

        # Механика \"волны\": нужно собрать ВСЕ камни
        self.wave_in_progress = False     # Флаг, идет ли сейчас волна
        self.wave_size = 0               # Сколько камней в волне
        self.wave_collected = 0          # Сколько камней собрано
        self.wave_failed = False         # Пропущен ли хоть один камень

        # Очки
        self.score = 0
        self.max_possible_score = 0  # Сколько всего можно набрать
        self.wave_processed = False

    
        # Таймеры
        self.start_time = None
        self.airborne_time = 0  # Время в воздухе
        self.is_airborne = False  # В воздухе ли персонаж?
        self.airborne_start_time = None  # Когда начался полет

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
        self.max_possible_score = 0

        # Время старта
        self.start_time = time.time()
        self.last_wave_time = self.start_time
        self.airborne_time = 0
        self.is_airborne = False
        self.airborne_start_time = None

        # Генератор голосового анализа
        self.voice_generator = analyze_voice()
        self.start_new_wave()

    def on_draw(self):
        arcade.start_render()
        # Рисуем фон
        self.background.draw()
        # Рисуем камни
        self.gem_list.draw()
        #print("Нарисовали камень")
        # Рисуем игрока
        self.player.draw()

        
        # Очки и время
        arcade.draw_text(f"Score: {self.score}", 10, SCREEN_HEIGHT - 30, arcade.color.BLACK, 20)
        arcade.draw_text(f"Time left: {int(GAME_DURATION - (time.time() - self.start_time))}", 
                         SCREEN_WIDTH - 120, SCREEN_HEIGHT - 30, arcade.color.BLACK, 20)
        
        # **Реальный таймер в воздухе**
        arcade.draw_text(f"Air time: {self.airborne_time:.1f}s", 
                         SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30, arcade.color.BLACK, 20, anchor_x="center")

        if time.time() - self.start_time > GAME_DURATION:
            arcade.draw_text("GAME OVER!", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2,
                             arcade.color.RED, 40, anchor_x="center")
            arcade.draw_text(f"Final Score: {self.score} / {self.max_possible_score}",
                             SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50,
                             arcade.color.WHITE, 30, anchor_x="center")
            
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

        # **Обновляем таймер воздуха**
        if self.player.center_y > GROUND_Y:
            if not self.is_airborne:
                self.is_airborne = True
                self.airborne_start_time = time.time()  # Фиксируем время старта
            else:
                self.airborne_time = time.time() - self.airborne_start_time  # Обновляем в реальном времени
        else:
            if self.is_airborne:
                self.is_airborne = False
                self.airborne_time = 0  # Сбрасываем таймер


        # Обновляем анимацию персонажа
        self.player.update_animation(delta_time)

        #print(f"Количество камней перед проверкой: {len(self.gem_list)}")

        # Проверка столкновений
        for gem in self.gem_list:
            # Если этот камень ушел за экран - считаем, что волна провалена
            
            # Если столкнулись с игроком
            if arcade.check_for_collision(self.player, gem):
                gem.remove_from_sprite_lists()
                self.wave_collected += 1
        
        
            if gem.right < 10:
                gem.remove_from_sprite_lists()
                self.wave_failed = True

        # Если волна активна и все камни пропали (или собраны)
        if self.wave_in_progress and not self.wave_processed and (self.wave_collected == self.wave_size or self.wave_failed):
            if not self.wave_failed:
                self.score += 10  # Очки начисляются только если все камни собраны
            self.wave_in_progress = False  # Завершаем волну
            # self.wave_processed = True
            self.start_new_wave()

        # # Запускаем новую волну, если прошло время и предыдущая волна завершена
        # if (time.time() - self.last_wave_time >= WAVE_INTERVAL) and self.wave_in_progress==False:
        #     self.wave_processed = False
            

    def start_new_wave(self):
        """
        Создаем \"волну\" из нескольких камней сразу,
        чтобы игрок получил очки только при сборе всех.
        """
        self.wave_in_progress = True
        self.wave_failed = False
        self.wave_collected = 0
        self.wave_size = WAVE_SIZE  # Теперь self.wave_size обновляется перед каждой волной
        self.last_wave_time = time.time()  # Запоминаем время создания волны
        self.gem_wave_time =  time.time()
        self.gem_count=0
        self.max_possible_score += 10  # Теперь 100% обновляется при создании волны
        
       
    
   
        arcade.schedule(self.add_gem, COLLECTION_TIME / WAVE_SIZE)
       
    
    def add_gem(self, delta_time=0):
        """Добавляет один камень с задержкой"""
        if self.gem_count < self.wave_size:
            gem_x = SCREEN_WIDTH + 100
            fixed_y = AIR_Y
            new_gem = Gem(gem_x, fixed_y)
            self.gem_list.append(new_gem)
            print(f"Добавлен камень {self.gem_count + 1}/{self.wave_size}")

            self.gem_count += 1

        # Если все камни добавлены, отключаем таймер
        if self.gem_count >= self.wave_size:
            arcade.unschedule(self.add_gem)

# ------------------------- Запуск -------------------------
if __name__ == "__main__":
    game = VoiceArcadeGame()
    
    game.setup()
    arcade.run()
