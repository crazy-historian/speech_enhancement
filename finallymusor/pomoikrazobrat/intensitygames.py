import arcade
import random
import time
import numpy as np
import parselmouth
import os
from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ------------------------- Глобальные настройки -------------------------
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 900
SCREEN_TITLE = "Voice-Controlled Arcade Game"

SCROLL_SPEED = 25
low_quiet, high_quiet = 40, 45
low_norm, high_norm = 45, 50
low_loud, high_loud = 60 ,65
chastota=6
GROUND_Y = 100
AIR_Y = 800

GAME_DURATION = 60

BLOCKSIZE = 1024
SILENCE_THRESHOLD_DB = 40.0
BLOCKS_TO_SILENT = 4

SMOOTHING_ALPHA = 0.3  # Коэффициент экспоненциального сглаживания
RESPONSE_FACTOR = 0.9  # Коэффициент быстрого отклика

ARTIFACT_SCORE = 10
ARTIFACT_DURATION = 2.0  # Время, за которое персонаж должен полностью пролететь текстуру (в секундах)
CHECK_SCORE=0.7
# Пути к текстурам заданий
TEXTURE_ONE = "tekstureone.png"
TEXTURE_TWO = "teksturetoo.png"

# ------------------------- Функция анализа интенсивности голоса -------------------------
def analyze_voice():
    with InputStream(samplerate=16000, blocksize=BLOCKSIZE, channels=1, sampwidth=2) as stream:
        stream.set_methods(UnpackRawInFloat32())
        start_time = time.time()
        while time.time() - start_time < GAME_DURATION:
            raw_data = stream.read(BLOCKSIZE)
            if not raw_data:
                continue
            signal = stream.chain_of_methods(raw_data)
            sound = parselmouth.Sound(values=signal, sampling_frequency=stream.samplerate)

            intensity_obj = sound.to_intensity()
            intensity_values = intensity_obj.values.T.flatten()
            avg_intensity = np.mean(intensity_values) if len(intensity_values) > 0 else 0
            yield avg_intensity

# ------------------------- Класс персонажа -------------------------
class PlayerCharacter(arcade.Sprite):
    def __init__(self):
        super().__init__()
        self.texture = arcade.load_texture(":resources:images/animated_characters/male_person/malePerson_idle.png")
        self.scale = 0.5
        self.center_x = 150
        self.center_y = GROUND_Y
        self.target_y = GROUND_Y

    def update_position(self, intensity: float):
        target_position = GROUND_Y + (intensity / 100) * (AIR_Y - GROUND_Y)
        self.target_y = SMOOTHING_ALPHA * target_position + (1 - SMOOTHING_ALPHA) * self.target_y

        # Быстрое реагирование на большие изменения
        if abs(self.center_y - self.target_y) > 20:
            self.center_y += (self.target_y - self.center_y) * RESPONSE_FACTOR
        else:
            self.center_y = self.target_y

        self.center_y = max(GROUND_Y, min(self.center_y, AIR_Y))

# ------------------------- Класс задания (Artifact) -------------------------
# ------------------------- Класс задания (Artifact) -------------------------
class Artifact(arcade.Sprite):
    def __init__(self, intensity):
        super().__init__(TEXTURE_ONE, scale=0.2)
        self.center_x = SCREEN_WIDTH + random.randint(50, 300)
        self.center_y = GROUND_Y + (intensity / 100) * (AIR_Y - GROUND_Y)
        self.change_x = -SCROLL_SPEED
        self.collected = False  # Флаг, был ли персонаж внутри задания
        self.inside_time = 0.0  # Время нахождения внутри задания
        self.last_time_inside = None

        # Определение громкости и соответствующего текста
        if low_norm <= intensity <= high_norm:
            self.volume_label = "Нормально"
        elif low_loud <= intensity <= high_loud:
            self.volume_label = "Громко"
        elif low_quiet <= intensity <= high_quiet:
            self.volume_label = "Неизвестно"

        # Длина текстуры рассчитывается так, чтобы персонаж пролетал ее за ARTIFACT_DURATION секунд
        self.width = SCROLL_SPEED * ARTIFACT_DURATION * 10  

    def update(self):
        self.center_x += self.change_x
        if self.right < 0:
            self.remove_from_sprite_lists()

    def check_collision(self, player, delta_time):
        if arcade.check_for_collision(self, player):
            self.texture = arcade.load_texture(TEXTURE_TWO)
            self.width = SCROLL_SPEED * ARTIFACT_DURATION * 10
            
            if self.last_time_inside is None:
                self.last_time_inside = time.time()
            else:
                self.inside_time += delta_time
            self.collected = True
        else:
            self.texture = arcade.load_texture(TEXTURE_ONE)
            self.width = SCROLL_SPEED * ARTIFACT_DURATION * 10
            self.last_time_inside = None

        if self.collected and self.inside_time>=CHECK_SCORE and player.center_x > self.right-10:
            self.remove_from_sprite_lists()
            return True  
     
        return False

# ------------------------- Класс игры -------------------------
class VoiceArcadeGame(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        self.player = None
        self.artifact_list = None
        self.score = 0
        self.start_time = None
        self.voice_generator = None
        self.last_artifact_time = 0
        self.current_intensity = 0
        self.total_artifacts = 0
        self.intensity_history = []  # История интенсивности
        self.artifact_intervals = []  # Хранение данных об артефактах
        arcade.set_background_color(arcade.csscolor.SKY_BLUE)

    def setup(self):
        self.player = PlayerCharacter()
        self.artifact_list = arcade.SpriteList()
        self.score = 0
        self.total_artifacts = 0
        self.start_time = time.time()
        self.voice_generator = analyze_voice()
        self.intensity_history = []  # История интенсивности
        self.artifact_intervals = []  # Сброс истории артефактов

    def on_draw(self):
        arcade.start_render()
        self.artifact_list.draw()
        self.player.draw()
        arcade.draw_text(f"Score: {self.score}", 10, SCREEN_HEIGHT - 30, arcade.color.BLACK, 20)
        elapsed = time.time() - self.start_time
        left = GAME_DURATION - elapsed
        arcade.draw_text(f"Time: {int(left)}", SCREEN_WIDTH - 120, SCREEN_HEIGHT - 30, arcade.color.BLACK, 20)
        
        if left <= 0:
            arcade.draw_text(f"Final Score: {self.score}/{self.total_artifacts * ARTIFACT_SCORE}", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, arcade.color.RED, 40, anchor_x="center")
            
            self.show_intensity_graph()

        for artifact in self.artifact_list:
            if artifact.collected:
                arcade.draw_text(f"{artifact.inside_time:.1f}", artifact.center_x - 20, artifact.center_y + 20, arcade.color.BLACK, 14)

            # Отображение громкости ("Громко" или "Нормально")
            arcade.draw_text(artifact.volume_label, artifact.center_x - 20, artifact.center_y + 40, arcade.color.RED, 14)

    def on_update(self, delta_time):
        elapsed = time.time() - self.start_time
        if elapsed > GAME_DURATION:
            return

        self.artifact_list.update()
        try:
            self.current_intensity = next(self.voice_generator)
        except StopIteration:
            self.current_intensity = 0

        self.intensity_history.append(self.current_intensity)  # Сохраняем значение интенсивности

        self.player.update_position(self.current_intensity)
        self.spawn_artifacts()

        for artifact in self.artifact_list:
            if artifact.check_collision(self.player, delta_time):
                self.score += ARTIFACT_SCORE
                # Сохранение данных об артефакте (начало, конец, интенсивность)
                self.artifact_intervals.append((elapsed, elapsed + ARTIFACT_DURATION, artifact.center_y))


    def spawn_artifacts(self):
        current_time = time.time()
        if current_time - self.last_artifact_time > chastota:
            intensity_range = random.choice([(low_norm, high_norm), (low_loud, high_loud)])
            intensity = random.randint(*intensity_range)
            self.artifact_list.append(Artifact(intensity))
            self.total_artifacts += 1
            self.last_artifact_time = current_time

    def show_intensity_graph(self):
        """Построить график интенсивности голоса с выделением артефактов"""
        time_axis = np.linspace(0, GAME_DURATION, len(self.intensity_history))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(time_axis, self.intensity_history, label="Voice Intensity", color="blue")

        # Добавление артефактов в виде прямоугольников
        for start, end, intensity in self.artifact_intervals:
            
            rect = patches.Rectangle((start, intensity/10 - 5), end - start+3, 10, linewidth=1,
                                    edgecolor="yellow", facecolor="yellow", alpha=0.5)
            ax.add_patch(rect)

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Intensity")
        ax.set_title("Voice Intensity Over Time")
        ax.legend()
        ax.grid(True)
        plt.show()



if __name__ == "__main__":
    game = VoiceArcadeGame()
    game.setup()
    arcade.run()