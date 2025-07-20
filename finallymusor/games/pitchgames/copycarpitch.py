import arcade
import time
import parselmouth
import numpy as np
from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32

# =========================
# --- Глобальные настройки
# =========================
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 512
SCREEN_TITLE = "Voice Ship Control"
TRACK_CENTER_Y = SCREEN_HEIGHT - 350
TRACK_HEIGHT = 150
TRACK_TOP = TRACK_CENTER_Y + TRACK_HEIGHT // 2
TRACK_BOTTOM = TRACK_CENTER_Y - TRACK_HEIGHT // 2

# Параметры движения
START_X = 100
END_X = 1000
SHIP_SPEED = (END_X - START_X) / 2

# Зона pitch
PITCH_MIN = 180
PITCH_MAX = 240
SMOOTHING_FACTOR = 0.2  # сглаживание движения по Y

# Аудио
BLOCKSIZE = 1024
SILENCE_THRESHOLD_DB = 50.0
BLOCKS_TO_SILENT = 4
PITCH_FLOOR = 100
PITCH_CEILING = 600
VOICING_THRESHOLD = 0.6

# Максимальное пребывание за пределами
MAX_OUT_OF_ZONE_DURATION = 0.5

# =========================
# --- Класс скроллируемого фона
# =========================
class ScrollingBackground:
    """
    Отвечает за 3 текстуры фона:
    1) bg_1 — стартовая (state='start')
    2) bg_2 — основная (state='main')
    3) bg_3 — используется в "челлендже", когда запущено задание (state='challenge'),
       затем вернёмся к bg_2.

    Параметры:
      - textures: [bg_1, bg_2, bg_3]
      - scroll_speed: скорость скролла пикселей в секунду
    """
    def __init__(self, textures, scroll_speed):
        self.textures = textures  # [bg_1, bg_2, bg_3]
        self.scroll_speed = scroll_speed
        self.active = []  # список словарей {"texture": ..., "x": ...}
        self.state = "start"
        self.challenge_active = False
        self.challenge_end_time = None
        self.reset()

    def reset(self):
        self.active = [{"texture": self.textures[0], "x": 0}]
        self.state = "start"
        self.challenge_active = False
        self.challenge_end_time = None

    def start_challenge(self, duration):
        """
        Запускает "задание" (челлендж) на указанное время.
        Текстура bg_3 добавляется так, чтобы её правый край был в x=100
        к моменту окончания задания.
        """
        self.state = "challenge"
        self.challenge_active = True
        self.challenge_end_time = time.time() + duration

            # Куда должен попасть правый край bg_3
        desired_right_x = 100
        required_offset = self.scroll_speed * duration
        start_x_bg3 = desired_right_x + required_offset - SCREEN_WIDTH

        # Где сейчас заканчивается последний фон
        right_edge = max(t["x"] + SCREEN_WIDTH for t in self.active)

        # 🔧 Добавим bg_2 до bg_3, если есть зазор
        while right_edge < start_x_bg3:
            self.active.append({
                "texture": self.textures[1],
                "x": right_edge
            })
            right_edge += SCREEN_WIDTH

        # ✅ Вставляем bg_3
        self.active.append({
            "texture": self.textures[2],
            "x": start_x_bg3
        })

        # ✅ И сразу за ним — bg_2, чтобы не было дыры после челленджа
        self.active.append({
            "texture": self.textures[1],
            "x": start_x_bg3 + SCREEN_WIDTH
        })

    def update(self, delta_time, voice_active, pitch_in_range):
        """
        Обновляет позиции текстур, если есть голос (voice_active).
        - Если state=='start', при первом голосе + pitch_in_range переходим к bg_2
        - Если challenge_active, добавляем bg_3
        - Иначе крутим bg_2 бесшовно
        """
        # Если нет голоса — фон "застывает"
        if not voice_active:
            return

        scroll_amount = self.scroll_speed * delta_time
        for tex in self.active:
            tex["x"] -= scroll_amount

        # Удаляем текстуры, которые ушли целиком за левую границу экрана
        self.active = [t for t in self.active if t["x"] + SCREEN_WIDTH > 0]

        # ---------------------------
        # --- Основной скроллинг
        # ---------------------------
        # Если не челлендж и не старт, крутим bg_2 циклично
        if not self.challenge_active and self.state != "start":
            # Если текстур меньше 2 (не хватает на покрытие экрана), добавим новую bg_2
            if len(self.active) < 2:
                right_edge = max(t["x"] + SCREEN_WIDTH for t in self.active)
                self.active.append({"texture": self.textures[1], "x": right_edge})

        # ---------------------------
        # --- Завершение челленджа
        # ---------------------------
        if self.challenge_active and time.time() >= self.challenge_end_time:
            self.challenge_active = False
            self.state = "main"
            right_edge = max(t["x"] + SCREEN_WIDTH for t in self.active)
            self.active.append({"texture": self.textures[1], "x": right_edge})

        # ---------------------------
        # --- Переход со старта на основной
        # ---------------------------
        if self.state == "start" and voice_active and pitch_in_range:
            self.state = "main"
            last_x = self.active[-1]["x"]
            self.active.append({"texture": self.textures[1], "x": last_x + SCREEN_WIDTH})

    def draw(self):
        # Рисуем каждую "полоску" фона так, чтобы её центр был по центру экрана по Y
        for tex in self.active:
            arcade.draw_texture_rectangle(
                tex["x"] + SCREEN_WIDTH // 2,
                SCREEN_HEIGHT // 2,
                SCREEN_WIDTH,
                SCREEN_HEIGHT,
                tex["texture"]
            )

# =========================
# --- Корабль
# =========================
class Ship(arcade.Sprite):
    def __init__(self):
        super().__init__("cargame/car.png", scale=0.25)
        self.center_x = START_X
        self.center_y = SCREEN_HEIGHT -350
        self.moving_forward = False
        self.out_of_zone = False
        self.rotating = False
        self.rotation_speed = 300  # градусов в секунду

        # Загружаем все текстуры
        self.frames = [
            arcade.load_texture(f"cargame/car_{i}.png") for i in range(1, 5)
        ]
        self.current_frame = 0
        self.animation_timer = 0.0
        self.frame_duration = 0.1  # сколько секунд показывать один кадр
    def start_rotation(self):
        self.rotating = True
    
    def update_rotation(self, delta_time):
        if self.rotating:
            self.angle += self.rotation_speed * delta_time
    
    def update_animation(self, delta_time: float):
        self.animation_timer += delta_time
        if self.animation_timer >= self.frame_duration:
            self.animation_timer = 0.0
            self.current_frame = (self.current_frame + 1) % len(self.frames)
            self.texture = self.frames[self.current_frame]

    def update_position(self, voice_status, pitch, delta_time):
        """
        Двигаем корабль по логике:
          - Если есть голос и pitch в зоне (PITCH_MIN..PITCH_MAX) → двигаем по Y в рамках зелёной области
          - Если есть голос, но вышел из зоны → всё равно двигаем по X, Y растягиваем к "красным" зонам
          - Если голоса нет → корабль останавливается, Y тянется к центру
        """
        if voice_status == 1 and PITCH_MIN <= pitch <= PITCH_MAX:
            self.moving_forward = True

            pitch_range = PITCH_MAX - PITCH_MIN
            relative_position = (pitch - PITCH_MIN) / pitch_range

            # Вместо SCREEN_HEIGHT // 2 используем TRACK_CENTER_Y
            target_y = TRACK_CENTER_Y - TRACK_HEIGHT // 2 + relative_position * TRACK_HEIGHT
            self.center_y += (target_y - self.center_y) * SMOOTHING_FACTOR

        elif voice_status == 1:
            self.moving_forward = True
            direction = "up" if pitch > PITCH_MAX else "down"
            overshoot = abs(pitch - (PITCH_MAX if direction == "up" else PITCH_MIN)) / 100
            offset = overshoot * (TRACK_CENTER_Y - 100)

            if direction == "up":
                target_y = TRACK_CENTER_Y + TRACK_HEIGHT // 2 + offset
            else:
                target_y = TRACK_CENTER_Y - TRACK_HEIGHT // 2 - offset

            self.center_y += (target_y - self.center_y) * SMOOTHING_FACTOR

        else:
            self.moving_forward = False
            self.center_y += (TRACK_CENTER_Y - self.center_y) * 0.1

        # if self.moving_forward and self.center_x < END_X:
        #     self.center_x += SHIP_SPEED * delta_time

# =========================
# --- Основной класс игры
# =========================
class VoiceShipGame(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)

        arcade.set_background_color(arcade.color.SKY_BLUE)
        self.ship = Ship()

        # Подключаем анализ голоса
        self.voice_generator = analyze_voice()
        self.current_pitch = None
        self.rotation_start_time = None


        # Флаги состояния
        self.started = False
        self.game_over = False
        self.success = False

        # Таймер вне зоны
        self.out_of_zone_timer = 0.0

        # Таймер общего прохождения
        self.timer_start_time = None
        self.elapsed_time = 0

        # Подгружаем текстуры фона и создаём контроллер скроллинга
        self.bg_1 = arcade.load_texture("cargame/carbg_1.png")
        self.bg_2 = arcade.load_texture("cargame/carbg_2.png")
        self.bg_3 = arcade.load_texture("cargame/carbg_3.png")
        self.scrolling_background = ScrollingBackground(
            [self.bg_1, self.bg_2, self.bg_3],
            scroll_speed=500  # пикселей в секунду
        )

    def reset_game(self):
        """
        Сбрасывает игру до начальных значений,
        включая фон, корабль и голосовой анализ.
        """
        self.ship = Ship()
        self.voice_generator = analyze_voice()
        self.current_pitch = None
        self.started = False
        self.game_over = False
        self.success = False
        self.out_of_zone_timer = 0.0
        self.timer_start_time = None
        self.elapsed_time = 0
        self.rotation_start_time = None

        self.scrolling_background.reset()

    def on_draw(self):
        arcade.start_render()

        # 1) Рисуем фон
        self.scrolling_background.draw()

        # 2) Рисуем корабль
        self.ship.draw()

        # 3) Зоны
        arcade.draw_rectangle_outline(SCREEN_WIDTH // 2, SCREEN_HEIGHT -350, SCREEN_WIDTH, 150, arcade.color.GREEN, 2)
        #arcade.draw_rectangle_outline(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 80, SCREEN_WIDTH, 160, arcade.color.RED, 2)
        #arcade.draw_rectangle_outline(SCREEN_WIDTH // 2, 80, SCREEN_WIDTH, 160, arcade.color.RED, 2)

        # 4) Победа
        if self.success and not self.game_over:
            arcade.draw_text(
                "SUCCESS!",
                SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2,
                arcade.color.GREEN, 50, anchor_x="center"
            )
            

        # 5) Проигрыш
        if self.game_over and not self.success:
            arcade.draw_text(
                "GAME OVER",
                SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2,
                arcade.color.RED, 50, anchor_x="center"
            )

        # 6) Таймер общего прохождения
        if self.timer_start_time:
            arcade.draw_text(
                f"Time: {self.elapsed_time:.2f}s",
                SCREEN_WIDTH // 2,
                SCREEN_HEIGHT - 60,
                arcade.color.BLACK, 20,
                anchor_x="center"
            )

        # 7) Шкала pitch
        self.draw_pitch_scale()

        # 8) Если челлендж активен, покажем таймер (сколько осталось)
        if self.scrolling_background.challenge_active:
            time_left = self.scrolling_background.challenge_end_time - time.time()
            if time_left < 0:
                self.success=True
                time_left = 0
                
            arcade.draw_text(
                f"Challenge: {time_left:.2f}s",
                10, SCREEN_HEIGHT - 30,
                arcade.color.RED, 20
            )

    def draw_pitch_scale(self):
        scale_x = SCREEN_WIDTH - 50
        top = SCREEN_HEIGHT - 20
        bottom = 20
        height = top - bottom

        # Ось
        arcade.draw_line(scale_x, bottom, scale_x, top, arcade.color.BLACK, 2)

        # Разметка (5 делений)
        for i in range(6):
            freq = PITCH_FLOOR + i * (PITCH_CEILING - PITCH_FLOOR) / 5
            y = bottom + (i / 5) * height
            arcade.draw_line(scale_x - 10, y, scale_x + 10, y, arcade.color.BLACK)
            arcade.draw_text(f"{int(freq)}", scale_x + 15, y - 8, arcade.color.BLACK, 12)

        # Плашка желаемого диапазона (PITCH_MIN..PITCH_MAX)
        y_min = bottom + ((PITCH_MIN - PITCH_FLOOR) / (PITCH_CEILING - PITCH_FLOOR)) * height
        y_max = bottom + ((PITCH_MAX - PITCH_FLOOR) / (PITCH_CEILING - PITCH_FLOOR)) * height
        arcade.draw_rectangle_filled(
            scale_x, (y_min + y_max) / 2,
            20, y_max - y_min,
            arcade.color.ALMOND
        )

        # Текущий pitch
        if self.current_pitch:
            y_current = bottom + ((self.current_pitch - PITCH_FLOOR) / (PITCH_CEILING - PITCH_FLOOR)) * height
            if bottom <= y_current <= top:
                arcade.draw_line(scale_x - 15, y_current, scale_x + 15, y_current, arcade.color.RED, 3)

    def on_update(self, delta_time: float):
        # Если уже конец игры или победа — ничего не делаем
        if self.game_over or self.success:
            return

        # Пытаемся получить (voice_status, pitch) от генератора
        try:
            voice_status, pitch = next(self.voice_generator)
        except StopIteration:
            return

        # Инициализация начала игры
        if not self.started and voice_status == 1:
            self.started = True
            self.scrolling_background.start_challenge(duration=4.0)

        # Если не началась — не обновляем корабль и прочее
        if not self.started:
            return

        # Запуск общего таймера, когда появляется голос
        if self.timer_start_time is None and voice_status == 1:
            self.timer_start_time = time.time()
            

        # Обновляем общее время
        if self.timer_start_time and not self.game_over and self.ship.center_x < END_X:
            self.elapsed_time = time.time() - self.timer_start_time

        self.current_pitch = pitch

        # Обновляем фон
        self.scrolling_background.update(
            delta_time,
            voice_active=(voice_status == 1),
            pitch_in_range=(PITCH_MIN <= pitch <= PITCH_MAX)
        )

        # Пример: запускаем челлендж на 4 секунды, если корабль уже проехал x=600, а челлендж ещё не активен
        # if (not self.scrolling_background.challenge_active) and (self.ship.center_x > 600):
        #     self.scrolling_background.start_challenge(duration=4.0)
        

        # Движение корабля
        self.ship.update_position(voice_status, pitch, delta_time)
        self.ship.update_rotation(delta_time)

        if not self.game_over:
            self.ship.update_animation(delta_time)

        if not self.success and not self.game_over and self.scrolling_background.challenge_end_time and not self.ship.rotating:
            if time.time() >= self.scrolling_background.challenge_end_time:
                self.success = True
                
                return
        
        if voice_status == 0:
            self.game_over = True
            return

        # Если вышли за границы зоны
        if self.ship.center_y < TRACK_BOTTOM or self.ship.center_y > TRACK_TOP:
            self.out_of_zone_timer += delta_time
            if self.out_of_zone_timer >= MAX_OUT_OF_ZONE_DURATION:
                if not self.ship.rotating:
                    self.ship.start_rotation()
                    self.rotation_start_time = time.time()
        else:
            self.out_of_zone_timer = 0.0
        
        if self.ship.rotating and not self.game_over and self.rotation_start_time:
            if time.time() - self.rotation_start_time >= 1.0:
                self.game_over = True


    def on_key_press(self, key, modifiers):
        # Перезапуск игры на клавишу R
        if key == arcade.key.R:
            self.reset_game()

# =========================
# --- Анализ голоса
# =========================
def analyze_voice():
    """
    Генератор (voice_status, pitch).
    voice_status: 1, если есть голос, 0 — нет голоса.
    pitch — частота, если есть (или последняя зафиксированная при недавнем голосе).
    """
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
            pitch_values = sound.to_pitch_ac(
                pitch_floor=PITCH_FLOOR,
                pitch_ceiling=PITCH_CEILING,
                voicing_threshold=VOICING_THRESHOLD
            ).selected_array['frequency']
            pitch_values[(pitch_values == 0) | (pitch_values > PITCH_CEILING)] = np.nan
            pitch = np.nanmean(pitch_values) if np.any(~np.isnan(pitch_values)) else last_valid_pitch

            if intensity > SILENCE_THRESHOLD_DB and pitch is not None:
                # Есть голос
                last_valid_pitch = pitch
                silent_counter = 0
                yield 1, pitch
            else:
                # Голоса нет (либо тихо, либо pitch ещё не появился)
                silent_counter += 1
                if last_valid_pitch is not None:
                    # Выдаём "1, last_valid_pitch" короткое время, если голос только что пропал
                    yield 1, last_valid_pitch
                else:
                    # Если никогда не было pitch — совсем 0, None
                    yield 0, None

                # Если некоторое число блоков подряд нет голоса — стабильно 0
                if silent_counter >= BLOCKS_TO_SILENT:
                    yield 0, last_valid_pitch

# =========================
# --- Точка входа
# =========================
if __name__ == "__main__":
    game = VoiceShipGame()
    arcade.run()
