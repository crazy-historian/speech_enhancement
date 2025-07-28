import arcade
import threading
import time
from pathlib import Path
from game.settings import (
    PITCH_CEILING, PITCH_FLOOR, SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE,
    PITCH_MIN, PITCH_MAX, TRACK_BOTTOM, TRACK_TOP,
    MAX_OUT_OF_ZONE_DURATION, END_X, START_X,
    SILENCE_THRESHOLD_DB, DURATION, SHIP_SPEED
)
from game.car import Car
from game.background import ScrollingBackground
from game.voice import analyze_voice


class VoiceCarGame(arcade.Window):
    def __init__(self, challenge_duration=3.0, profile=None, task=None):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)

        arcade.set_background_color(arcade.color.SKY_BLUE)
        self.ship = Car()

        # Подключаем анализ голоса
        self.voice_generator = analyze_voice()
        self.current_pitch = None
        self.voice_thread_running = True
        self.current_voice_status = 0
        self.rotation_start_time = None
        self.challenge_duration = challenge_duration
        self.profile = profile
        self.task = task
        threading.Thread(target=self.voice_loop, daemon=True).start()
        

        if profile and task:
            self.apply_task_settings(task)
        
        self.task_text_sprite = None
        if self.task and self.task.description:
            font_path = str(Path("games/voice_car/components/shrift_mult.otf").absolute())
            self.task_text_sprite = arcade.create_text_sprite(
                text=self.task.description,
                start_x=SCREEN_WIDTH // 2,
                start_y=SCREEN_HEIGHT // 2 + 100,
                font_size=50,
                font_name=font_path,
                color=arcade.color.RED_DEVIL,
                anchor_x="center",
                anchor_y="center"
    )

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
        self.bg_1 = arcade.load_texture("games/voice_car/components/carbg_1.png")
        self.bg_2 = arcade.load_texture("games/voice_car/components/carbg_2.png")
        self.bg_3 = arcade.load_texture("games/voice_car/components/carbg_3.png")
        scroll_speed = 800 if challenge_duration <= 2.5 else 500
        self.scrolling_background = ScrollingBackground(
            [self.bg_1, self.bg_2, self.bg_3],
            scroll_speed=scroll_speed
        )
    def voice_loop(self):
        for voice_status, pitch in analyze_voice():
            if not self.voice_thread_running:
                break
            self.current_voice_status = voice_status
            self.current_pitch = pitch
    
    def on_close(self):
        self.voice_thread_running = False
        super().on_close()
    def reset_game(self, task=None):
        """
        Сбрасывает игру до начальных значений,
        включая фон, корабль и голосовой анализ.
        """
        self.ship = Car()
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
        self.challenge_duration = self.challenge_duration
        if task:
            self.task = task
            self.apply_task_settings(task)
        
        if self.task and self.task.description:
            font_path = str(Path("games/voice_car/components/shrift_mult.otf").absolute())
            self.task_text_sprite = arcade.create_text_sprite(
                text=self.task.description,
                start_x=SCREEN_WIDTH // 2,
                start_y=SCREEN_HEIGHT // 2 + 100,
                font_size=50,
                font_name=font_path,
                color=arcade.color.RED_DEVIL,
                anchor_x="center",
                anchor_y="center"
            )
        else:
            self.task_text_sprite = None
        
        
    def on_draw(self):
        arcade.start_render()

        # 1) Рисуем фон
        self.scrolling_background.draw()

        # 2) Рисуем корабль
        self.ship.draw()

        # 3) Зоны
        #arcade.draw_rectangle_outline(SCREEN_WIDTH // 2, SCREEN_HEIGHT -350, SCREEN_WIDTH, 150, arcade.color.GREEN, 2)
        #arcade.draw_rectangle_outline(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 80, SCREEN_WIDTH, 160, arcade.color.RED, 2)
        #arcade.draw_rectangle_outline(SCREEN_WIDTH // 2, 80, SCREEN_WIDTH, 160, arcade.color.RED, 2)

        # if self.task and self.task.description:
        #     arcade.draw_text(
        #     self.task.description,
        #     SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 100,  # немного выше центра
        #     arcade.color.PINK_LACE,
        #     40,
        #     width=SCREEN_WIDTH - 200,  # немного отступов слева и справа
        #     align="center",
        #     anchor_x="center",
        #     anchor_y="center",         # чтобы учитывать вертикальное центрирование
        #     multiline=True
        #     )
        
        if self.task_text_sprite:
            self.task_text_sprite.draw()

        
        # 4) Победа
        if self.success and not self.game_over:
            arcade.draw_text(
                "МОЛОДЕЦ!",
                SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2,
                arcade.color.TEA_GREEN, 50, anchor_x="center"
            )
            

        # 5) Проигрыш
        if self.game_over and not self.success:
            arcade.draw_text(
                "Попробуй еще раз",
                SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2,
                arcade.color.YELLOW_ORANGE, 50, anchor_x="center"
            )

        # 6) Таймер общего прохождения
        # if self.timer_start_time:
        #     arcade.draw_text(
        #         f"Time: {self.elapsed_time:.2f}s",
        #         SCREEN_WIDTH // 2,
        #         SCREEN_HEIGHT - 60,
        #         arcade.color.BLACK, 20,
        #         anchor_x="center"
        #     )

        # 7) Шкала pitch
        #self.draw_pitch_scale()

        # 8) Если челлендж активен, покажем таймер (сколько осталось)
        # if self.scrolling_background.challenge_active:
        #     time_left = self.scrolling_background.challenge_end_time - time.time()
        #     if time_left < 0:
        #         self.success=True
        #         time_left = 0
                
        #     arcade.draw_text(
        #         f"Challenge: {time_left:.2f}s",
        #         10, SCREEN_HEIGHT - 30,
        #         arcade.color.RED, 20
        #     )
        

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
        voice_status = self.current_voice_status
        pitch = self.current_pitch
       
        # Инициализация начала игры
        if not self.started and voice_status == 1:
            self.started = True
            self.scrolling_background.start_challenge(duration=self.challenge_duration)


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
    

    def apply_task_settings(self, task):
        global PITCH_MIN, PITCH_MAX, SILENCE_THRESHOLD_DB, MAX_OUT_OF_ZONE_DURATION, DURATION, SHIP_SPEED

        if isinstance(task.frequency, str):
            pitch_range = self.profile.settings['pitch_ranges'].get(task.frequency)
            if pitch_range is None:
                raise ValueError(f"Не найден диапазон частот '{task.frequency}' в профиле.")
        else:
            pitch_range = task.frequency

        PITCH_MIN, PITCH_MAX = pitch_range
        SILENCE_THRESHOLD_DB = self.profile.settings.get("volume_threshold_db", 50.0)
        MAX_OUT_OF_ZONE_DURATION = task.max_out_of_zone
        DURATION = task.duration
        SHIP_SPEED = (END_X - START_X) / DURATION

        self.challenge_duration = DURATION


    def on_key_press(self, key, modifiers):
        # Перезапуск игры на клавишу R
        if key == arcade.key.R:
            self.reset_game()
        if key == arcade.key.T:
            from guicar import select_task_from_profile  # импортим прямо тут, чтобы избежать циклических импортов

            new_task = select_task_from_profile(self.profile)
            if new_task:
                print(f"Новое задание выбрано: {new_task.name}")
                self.apply_task_settings(new_task)
                self.reset_game(task=new_task)
