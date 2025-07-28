from game.settings import (SCREEN_HEIGHT, SCREEN_WIDTH, SCREEN_TITLE, 
                           ARTIFACTS_IN_WAVE, ARTIFACT_INTERVAL, chastota, GAME_DURATION, 
                           BLOCKSIZE, PITCH_CEILING, PITCH_FLOOR, ARTIFACT_SCORE, GROUND_Y, AIR_Y, CURRENT_TASK_TEXT, selected_ranges)
import arcade 
import time
import random
import numpy as np
from game.background import ScrollingBackground
from game.player import PlayerCharacter
import threading
from game.voice import analyze_voice
from game.artifact import ArtifactGroup


class VoiceArcadeGame(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        self.player = None
        self.artifact_groups = []
        self.artifact_list = arcade.SpriteList()
        self.voice_thread_running = True
        self.collision_counter = 0 #тоже видимо нигде не увеличивается
        self.score = 0
        self.start_time = None
        self.game_over = False
        self.current_pitch = None
        self.counter_spawn = [] #вроде нигде не используется
        self.wave_statuses = {} #вроде нигде не используется
        self.voice_active_start = None #вроде нигде не используется
        self.voice_active_time = 0.0 #вроде нигде не используется
        self.collision_flag = False #кажется, это тоже какая-то бесполезная переменная

        self.bg = ScrollingBackground("games/pichik/components/fon_bird3.png", speed=5)

        self.rating = 0 #посчитать рейтинг: кол-во очков
        self.final_text = "" #что выводится в конце игры
        self.kombo_sprite = arcade.Sprite("games/pichik/components/kombo.png", scale=0.2) #спрайт для вывода комбо
        self.kombo_sprite.center_x = SCREEN_WIDTH // 2 #спрайт для вывода комбо
        self.kombo_sprite.center_y = SCREEN_HEIGHT // 2 - 200 #спрайт для вывода комбо

        # Для волн
        self.in_wave = False
        self.artifacts_spawned = 0
        self.artifacts_in_wave = ARTIFACTS_IN_WAVE
        self.artifact_interval = ARTIFACT_INTERVAL
        self.time_since_last_artifact = 0
        self.wave_timer = 0

        self.total_artifacts = 0
        self.intensity_wave = 0  # здесь будем хранить "требуемую" чатстоту волны
        self.time_last_wave = 0 #это вероятно удалить можно
        self.pitch_history = [] 
        self.block_interval = BLOCKSIZE / 16000 #это вроде не используется

    def setup(self):
        self.player = PlayerCharacter()
        self.start_time = time.time()
        self.voice_generator = analyze_voice()
        # Генерируем первый wave_timer так, чтобы сразу можно было спавнить волну
        self.wave_timer = time.time() - chastota  

        # Запускаем поток чтения pitch
        threading.Thread(target=self.voice_loop, daemon=True).start()
       


    def voice_loop(self):
        for pitch in self.voice_generator:
            if not self.voice_thread_running:
                break
            self.current_pitch = pitch
            self.pitch_history.append(pitch if pitch is not None else np.nan)

    def on_close(self): #это лишняя функция, нигде не используется, закрытие происходит в on_update
        self.voice_thread_running = False
        super().on_close()

    def spawn_artifacts(self):
        now = time.time()
        if now - self.start_time > GAME_DURATION:
            return

        if not self.in_wave and (now - self.wave_timer >= chastota):
            self.in_wave = True
            self.artifacts_spawned = 0
            self.time_since_last_artifact = now

        if self.in_wave:
            if (now - self.time_since_last_artifact >= self.artifact_interval
                and self.artifacts_spawned < self.artifacts_in_wave):

                # Выбираем случайный диапазон из selected_ranges
                if self.artifacts_spawned == 0:
                    intensity_range = random.choice(selected_ranges)
                    self.intensity_wave = random.randint(*intensity_range)

                group = ArtifactGroup(CURRENT_TASK_TEXT, self.intensity_wave)
                group.add_to_list(self.artifact_list)
                self.artifact_groups.append(group)

                self.artifacts_spawned += 1
                self.total_artifacts += 1
                self.time_since_last_artifact = now

            if self.artifacts_spawned >= self.artifacts_in_wave:
                self.in_wave = False
                self.wave_timer = now

    
    def on_draw(self):
        arcade.start_render()
        self.bg.draw()
        for group in self.artifact_groups:
            for letter in group.letters:
                letter.draw()
        self.player.draw()

        self.draw_pitch_scale()

        arcade.draw_text(
            f"Score: {self.score}",
            10, SCREEN_HEIGHT - 30,
            arcade.color.BLACK,
            20
        )

        if self.game_over:
            # arcade.draw_text(
            #     self.final_text,
            #     SCREEN_WIDTH // 2,
            #     SCREEN_HEIGHT // 2 + 50,
            #     arcade.color.BLACK,
            #     font_size=30,
            #     anchor_x="center",
            #     anchor_y="center"
            # )
            stars_path = f"games/pichik/components/star_{self.rating}.png"
            texture = arcade.load_texture(stars_path)
            arcade.draw_texture_rectangle(
                SCREEN_WIDTH // 2,
                SCREEN_HEIGHT // 2 - 50,
                texture.width,
                texture.height,
                texture
            )
            #self.kombo_sprite.draw()

    def on_update(self, delta_time):
        if self.game_over:
            return

        elapsed = time.time() - self.start_time
        if elapsed >= GAME_DURATION:
            self.game_over = True
            self.voice_thread_running = False

            max_score = self.total_artifacts * ARTIFACT_SCORE
            self.rating = round((self.score / max_score) * 5) if max_score > 0 else 0
            self.rating = max(1, min(self.rating, 5))
            self.final_text = f"Игра окончена!\nВы набрали {self.score} из {max_score} очков."
            return

        self.bg.update()
        self.player.update_animation(delta_time)
        self.player.update_position(self.current_pitch)

        self.artifact_list.update()
        self.spawn_artifacts()

        # Проверка коллизий
        for group in self.artifact_groups:
            if group.update_and_check(self.player, delta_time):
                self.score += ARTIFACT_SCORE

        # Счётчик волны
        if self.collision_counter == self.artifacts_in_wave:
            self.score += ARTIFACT_SCORE
            self.collision_counter = 0

        for artifact in self.artifact_list:
            if artifact.right < 150:
                self.collision_counter = 0
                self.collision_flag = True

    def draw_pitch_scale(self):
        scale_x = SCREEN_WIDTH - 50
        num_steps = 5
        min_pitch = PITCH_FLOOR
        max_pitch = PITCH_CEILING

        arcade.draw_line(scale_x, GROUND_Y, scale_x, AIR_Y, arcade.color.BLACK, 2)
        for i in range(num_steps + 1):
            pitch_val = int(min_pitch + i*(max_pitch - min_pitch)/num_steps)
            y_pos = GROUND_Y + (i/num_steps)*(AIR_Y - GROUND_Y)
            arcade.draw_line(scale_x - 10, y_pos, scale_x+10, y_pos, arcade.color.BLACK, 2)
            arcade.draw_text(f"{pitch_val} Hz", scale_x + 15, y_pos - 10, arcade.color.BLACK, 14)

        if self.current_pitch is not None:
            pitch_y = GROUND_Y + ((self.current_pitch - min_pitch)/(max_pitch - min_pitch))*(AIR_Y - GROUND_Y)
            pitch_y = max(GROUND_Y, min(pitch_y, AIR_Y))
            arcade.draw_line(scale_x - 20, pitch_y, scale_x + 20, pitch_y, arcade.color.RED, 4)
