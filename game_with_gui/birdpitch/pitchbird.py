# pitch_game.py
import arcade
import threading
import random
import time
import numpy as np
import parselmouth
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32
from collections import deque

# ------------------ Глобальные настройки, которые будем переопределять при запуске ------------------
SCREEN_WIDTH = 1276
SCREEN_HEIGHT = 660
SCREEN_TITLE = "Громик/Пичик"

SCROLL_SPEED = 10
GROUND_Y = 110
AIR_Y = 600

PITCH_FLOOR = 100
PITCH_CEILING = 300

BLOCKSIZE = 1024
SILENCE_THRESHOLD_DB = 50.0
BLOCKS_TO_SILENT = 3

SMOOTHING_ALPHA = 0.6
RESPONSE_FACTOR = 0.9

ARTIFACT_SCORE = 10

TEXTURE_ONE = "berries.png"

GAME_DURATION = 60
chastota = 6.0
ARTIFACTS_IN_WAVE = 5.0
ARTIFACT_INTERVAL = 0.2
selected_ranges = [(100, 140)]
CURRENT_TASK_TEXT = "ДА"
mic_device_index = None  # сюда будем подставлять устройство из конфига


# ------------------ Функция анализа голоса (Pitch) ------------------

def analyze_voice(window_size=3):
    with InputStream(
        samplerate=16000,
        blocksize=BLOCKSIZE,
        channels=1,
        sampwidth=2,
        device=mic_device_index
    ) as stream:
        stream.set_methods(UnpackRawInFloat32())
        silent_counter = 0
        start_time = time.time()
        last_valid_pitch = None  
        smoothing_window = deque(maxlen=window_size)  # окно сглаживания

        while time.time() - start_time < GAME_DURATION:
            raw_data = stream.read(BLOCKSIZE)
            if not raw_data:
                continue

            signal = stream.chain_of_methods(raw_data)
            sound = parselmouth.Sound(values=signal, sampling_frequency=stream.samplerate)

            intensity_obj = sound.to_intensity()
            intensity_values = intensity_obj.values.T.flatten()
            avg_intensity = np.mean(intensity_values) if len(intensity_values) > 0 else -50

            pitch_obj = sound.to_pitch_ac(
                time_step=0.01,
                pitch_floor=PITCH_FLOOR,
                pitch_ceiling=PITCH_CEILING,
                voicing_threshold=0.6
            )
            pitch_values = pitch_obj.selected_array["frequency"]
            pitch_values[(pitch_values == 0) | (pitch_values > 600)] = np.nan
            avg_pitch = np.nanmean(pitch_values) if np.any(~np.isnan(pitch_values)) else last_valid_pitch

            above_silence_threshold = avg_intensity > SILENCE_THRESHOLD_DB
            valid_pitch = avg_pitch is not None

            if above_silence_threshold and valid_pitch:
                silent_counter = 0
                last_valid_pitch = avg_pitch
                smoothing_window.append(avg_pitch)
                yield np.mean(smoothing_window)
            else:
                silent_counter += 1
                if silent_counter < BLOCKS_TO_SILENT and last_valid_pitch is not None:
                    smoothing_window.append(last_valid_pitch)
                    yield np.mean(smoothing_window)
                else:
                    yield None


# ------------------ Классы PlayerCharacter, ScrollingBackground, ArtifactGroup, VoiceArcadeGame ------------------
class PlayerCharacter(arcade.Sprite):
    def __init__(self, scale=0.15):
        super().__init__()
        self.textures = [arcade.load_texture(f"{i}_bird.png") for i in range(1, 7)]
        self.texture_index = 0
        self.set_texture(self.texture_index)

        self.scale = scale
        self.center_x = 150
        self.center_y = GROUND_Y
        self.target_y = GROUND_Y
        self.frame_timer = 0.0

    def update_animation(self, delta_time: float = 1/60):
        self.frame_timer += delta_time
        if self.frame_timer > 0.1:
            self.texture_index = (self.texture_index + 1) % len(self.textures)
            self.set_texture(self.texture_index)
            self.frame_timer = 0.0

    def update_position(self, pitch: float):
        if pitch is None:
            pitch = 100  # базовый “гравитационный” pitch

        # Преобразуем pitch [PITCH_FLOOR..PITCH_CEILING] в [GROUND_Y..AIR_Y]
        target_position = GROUND_Y + ((pitch - PITCH_FLOOR) / (PITCH_CEILING - PITCH_FLOOR)) * (AIR_Y - GROUND_Y)
        self.target_y = SMOOTHING_ALPHA * target_position + (1 - SMOOTHING_ALPHA) * self.target_y

        if abs(self.center_y - self.target_y) > 20:
            self.center_y += (self.target_y - self.center_y) * RESPONSE_FACTOR
        else:
            self.center_y = self.target_y

        self.center_y = max(GROUND_Y, min(self.center_y, AIR_Y))


class ScrollingBackground:
    def __init__(self, texture_path, speed):
        self.speed = speed
        texture = arcade.load_texture(texture_path)
        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT

        self.textures = [
            arcade.Sprite(texture_path, scale=1.0),
            arcade.Sprite(texture_path, scale=1.0)
        ]
        for sprite in self.textures:
            sprite.width = self.width
            sprite.height = self.height

        self.textures[0].center_x = SCREEN_WIDTH // 2
        self.textures[0].center_y = SCREEN_HEIGHT // 2

        self.textures[1].center_x = SCREEN_WIDTH + SCREEN_WIDTH // 2
        self.textures[1].center_y = SCREEN_HEIGHT // 2

    def update(self):
        for sprite in self.textures:
            sprite.center_x -= self.speed
            if sprite.right < 0:
                sprite.center_x += SCREEN_WIDTH * 2

    def draw(self):
        for sprite in self.textures:
            sprite.draw()


class ArtifactGroup:
    def __init__(self, task_text, pitch):
        self.letters = []
        spacing = 5  
        font_path = str(Path("alphabet/shrift_mult.otf").absolute())
        color=(165, 42, 42, 255)

        normalized = (pitch - PITCH_FLOOR) / (PITCH_CEILING - PITCH_FLOOR)
        normalized = max(0.0, min(normalized, 1.0))
        center_y = GROUND_Y + normalized * (AIR_Y - GROUND_Y)

        current_x = SCREEN_WIDTH + 100

        for letter in task_text:
            sprite = arcade.create_text_sprite(
                text=letter,
                start_x=current_x,
                start_y=center_y,
                font_size=48,
                font_name=font_path,
                color=color,
                anchor_x="center",
                anchor_y="center"
            )
            sprite.true_x = current_x
            sprite.center_y = center_y
            sprite.collected = False

            self.letters.append(sprite)
            current_x += sprite.width + spacing

        self.collected = False

    def add_to_list(self, sprite_list):
        for letter in self.letters:
            sprite_list.append(letter)

    def update_and_check(self, player, delta_time):
        if self.collected:
            return False
        all_collected = True
        for sprite in self.letters:
            if sprite.collected:
                continue

            sprite.true_x -= SCROLL_SPEED
            sprite.center_x = round(sprite.true_x)

            dx = abs(sprite.center_x - player.center_x)
            dy = abs(sprite.center_y - player.center_y)
            if dx <= sprite.width // 2 + 20 and dy <= sprite.height // 2 + 20:
                sprite.collected = True
                sprite.remove_from_sprite_lists()
            else:
                all_collected = False

        if all_collected:
            self.collected = True
            return True
        return False


class VoiceArcadeGame(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        self.player = None
        self.artifact_groups = []
        self.artifact_list = arcade.SpriteList()

        self.voice_thread_running = True
        self.collision_counter = 0
        self.score = 0
        self.start_time = None
        self.game_over = False

        self.current_pitch = None
        self.counter_spawn = []
        self.wave_statuses = {}
        self.voice_active_start = None
        self.voice_active_time = 0.0
        self.collision_flag = False

        self.bg = ScrollingBackground("fon_bird3.png", speed=5)

        self.rating = 0
        self.final_text = ""
        self.kombo_sprite = arcade.Sprite("stars/kombo.png", scale=0.2)
        self.kombo_sprite.center_x = SCREEN_WIDTH // 2
        self.kombo_sprite.center_y = SCREEN_HEIGHT // 2 - 200

        # Для волн
        self.in_wave = False
        self.artifacts_spawned = 0
        self.artifacts_in_wave = ARTIFACTS_IN_WAVE
        self.artifact_interval = ARTIFACT_INTERVAL
        self.time_since_last_artifact = 0
        self.wave_timer = 0

        self.total_artifacts = 0
        self.intensity_wave = 0  # здесь будем хранить "требуемую" громкость волны
        self.time_last_wave = 0
        self.pitch_history = []
        self.block_interval = BLOCKSIZE / 16000

    def setup(self):
        self.player = PlayerCharacter()
        self.start_time = time.time()
        self.voice_generator = analyze_voice()
        # Генерируем первый wave_timer так, чтобы сразу можно было спавнить волну
        self.wave_timer = time.time() - chastota  

        # Запускаем поток чтения pitch
        threading.Thread(target=self.voice_loop, daemon=True).start()
        threading.Timer(GAME_DURATION + 0.5, self.plot_pitch_graph).start()


    def voice_loop(self):
        for pitch in self.voice_generator:
            if not self.voice_thread_running:
                break
            self.current_pitch = pitch
            self.pitch_history.append(pitch if pitch is not None else np.nan)

    def on_close(self):
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

    def plot_pitch_graph(self):
        if not self.pitch_history:
            print("Нет данных для pitch-графика.")
            return

        timestamps = np.arange(len(self.pitch_history)) * self.block_interval
        history = np.array(self.pitch_history)

        plt.figure(figsize=(10, 4))
        plt.plot(timestamps, history, color='blue', label='Pitch (Hz)')
        plt.title("График частоты тона (Pitch) за всё время игры")
        plt.xlabel("Время (сек)")
        plt.ylabel("Pitch (Hz)")
        plt.ylim(100, 300)  # фиксируем шкалу по Y
        plt.yticks(np.arange(100, 301, 20))  # деления через 20 Гц
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        filename = f"pitch_plot_{time.strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename)
        plt.close()
        os.system(f"open {filename}")
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
            stars_path = f"stars/star_{self.rating}.png"
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


# ------------------ Функция-обёртка (как run_game_with_task) ------------------
def run_pitch_game_with_task(task):
    """
    Получает task-словарь из pitch_config.json, настраивает глобальные переменные,
    создаёт объект VoiceArcadeGame и запускает arcade.run().
    """
    global GAME_DURATION, ARTIFACTS_IN_WAVE, ARTIFACT_INTERVAL
    global chastota, selected_ranges, CURRENT_TASK_TEXT
    global mic_device_index, SMOOTHING_ALPHA, RESPONSE_FACTOR

    # Распакуем task
    GAME_DURATION = task.get("duration", 60)
    ARTIFACTS_IN_WAVE = float(task.get("artifacts_count", 5))
    ARTIFACT_INTERVAL = float(task.get("artifact_interval", 0.2))
    chastota = float(task.get("frequency", 6))
    CURRENT_TASK_TEXT = task.get("text", "ДА")

    # Включаем диапазоны (quiet/norm/loud) в selected_ranges
    selected_ranges = []
    if task.get("gen_quiet"):
        selected_ranges.append((task["quiet"], task["quiet"]))
    if task.get("gen_norm"):
        selected_ranges.append((task["norm"], task["norm"]))
    if task.get("gen_loud"):
        selected_ranges.append((task["loud"], task["loud"]))
    # fallback
    if not selected_ranges:
        selected_ranges = [(100, 120)]

    # Сглаживание
    if task.get("smooth", True):
        SMOOTHING_ALPHA = 0.6
        RESPONSE_FACTOR = 0.9
    else:
        SMOOTHING_ALPHA = 0.25
        RESPONSE_FACTOR = 0.7

    # Сохраним device_index, если есть
    mic_device_index = task.get("mic_device_index", None)

    # Запуск
    game = VoiceArcadeGame()
    game.setup()
    arcade.run()
