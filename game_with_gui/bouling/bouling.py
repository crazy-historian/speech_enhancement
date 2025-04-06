import arcade
import time
import numpy as np
import parselmouth
from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32
import threading

# ---------------- Настройки ----------------
SCREEN_WIDTH = 1020
SCREEN_HEIGHT = 512
SCREEN_TITLE = "Voice Bowling Game"
#текст на дорожке, рестарт далее/след уровень далее
#длительность дорожкой
#ускорение
BLOCKSIZE = 1024
PITCH_FLOOR = 100
PITCH_CEILING = 600
SILENCE_THRESHOLD_DB = 50.0
VOICING_THRESHOLD = 0.6
BLOCKS_TO_SILENT = 2


# ---------------- Анализ голоса ----------------
def analyze_voice(callback):
    def _listen():
        with InputStream(samplerate=16000, blocksize=BLOCKSIZE, channels=1, sampwidth=2) as stream:
            stream.set_methods(UnpackRawInFloat32())
            silent_counter = 0
            while True:
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
                    voicing_threshold=VOICING_THRESHOLD
                )
                pitch_values = pitch_obj.selected_array['frequency']
                pitch_values[(pitch_values == 0) | (pitch_values > PITCH_CEILING)] = np.nan
                avg_pitch = np.nanmean(pitch_values) if np.any(~np.isnan(pitch_values)) else np.nan

                above_silence_threshold = avg_intensity > SILENCE_THRESHOLD_DB
                valid_pitch = (avg_pitch > PITCH_FLOOR)
                if above_silence_threshold and valid_pitch:
                    silent_counter = 0
                    callback(1)
                else:
                    silent_counter += 1
                    if silent_counter >= BLOCKS_TO_SILENT:
                        callback(0)
    threading.Thread(target=_listen, daemon=True).start()


# ---------------- Игровое окно ----------------
class VoiceBowlingGame(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        self.background = None
        self.player = None
        self.ball = None
        self.pins = arcade.SpriteList()
        self.track=arcade.SpriteList()
        self.is_moving = False
        self.ball_speed = 0

        self.target_x = 800
        self.move_duration = 2
        self.total_distance = self.target_x - 110
        self.ball_speed = self.total_distance / (self.move_duration * 60)

        self.motion_start_time = None
        self.movement_timer = 0
        self.timer_frozen = False  # 👈 Флаг "заморозки" таймера

    def setup(self):
        arcade.set_background_color(arcade.color.LIGHT_BLUE)
        self.player = arcade.Sprite(":resources:images/animated_characters/male_person/malePerson_idle.png", scale=0.7)
        self.player.center_x = 100
        self.player.center_y = 150

        self.track=arcade.Sprite("fon_bouling_1.png", scale=1)
        self.track.center_x = 500
        self.track.center_y=140

        self.ball = arcade.Sprite("shar_bouling.png", scale=0.2)
        self.ball.center_x = 110
        self.ball.center_y = 150

        for i in range(5):
            pin = arcade.Sprite("keglya.png", scale=0.05)
            pin.center_x = 800 + i * 25
            pin.center_y = 145
            self.pins.append(pin)

        analyze_voice(self.on_voice_detected)

    def on_voice_detected(self, is_voiced):
        if is_voiced and not self.is_moving and not self.timer_frozen:
            self.motion_start_time = time.time()
        self.is_moving = bool(is_voiced)

    def on_draw(self):
        arcade.start_render()
        self.track.draw()

        self.player.draw()
        self.ball.draw()
        self.pins.draw()
        

        # 🕒 Отображение таймера (замороженного или активного)
        if self.motion_start_time and not self.timer_frozen:
            self.movement_timer = time.time() - self.motion_start_time

        arcade.draw_text(f"Ball time: {self.movement_timer:.2f} s",
                         SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30,
                         arcade.color.BLACK, 20, anchor_x="center")

    def on_update(self, delta_time: float):
        if self.is_moving and self.ball.center_x < self.target_x:
            self.ball.center_x += self.ball_speed
        elif self.is_moving and self.ball.center_x >= self.target_x:
            self.is_moving = False
            self.timer_frozen = True  # ⏸ Останавливаем таймер
        elif not self.is_moving:
            pass


# ---------------- Запуск ----------------
if __name__ == "__main__":
    game = VoiceBowlingGame()
    game.setup()
    arcade.run()
