import arcade
import time
import numpy as np
import parselmouth
from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32
import threading
from pathlib import Path

# ---------------- Настройки ----------------
SCREEN_WIDTH = 1020
SCREEN_HEIGHT = 512
SCREEN_TITLE = "Voice Bowling Game"

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

class SpeechTask(arcade.SpriteList):
    def __init__(self, text, center_x, center_y, letter_spacing=10, scale=0.3):
        super().__init__()
        self.text = text.upper()
        self.center_x = center_x
        self.center_y = center_y
        self.letter_spacing = letter_spacing
        self.scale = scale
        self._create_sprites()

    def _create_sprites(self):
        total_width = 0
        textures = []

        for letter in self.text:
            path = f"alphabet40/{letter} beresta.jpg"
            texture = arcade.load_texture(path)
            textures.append(texture)
            total_width += texture.width * self.scale + self.letter_spacing
            

        # начальная x координата для выравнивания по центру
        start_x = self.center_x - total_width / 2

        for i, texture in enumerate(textures):
            sprite = arcade.Sprite(texture=texture, scale=self.scale)
            sprite.center_x = start_x + sprite.width / 2 + i * (sprite.width + self.letter_spacing)
            sprite.center_y = self.center_y
            sprite.angle = -15
            self.append(sprite)

# ---------------- Класс Кегли ----------------
class Keglya(arcade.Sprite):
    def __init__(self, x, y):
        super().__init__("keglya/keglya_1.png", scale=0.7)
        self.textures = [
            arcade.load_texture("keglya/keglya_1.png"),
            arcade.load_texture("keglya/keglya_2.png"),
            arcade.load_texture("keglya/keglya_3.png")
        ]
        self.set_texture(0)
        self.hit_sound = arcade.load_sound("keglya/hit.wav")
        self.center_x = x
        self.center_y = y
        self.hit_count = 0
        self.fly = False
        self.fly_timer = 0.0

    def on_hit(self):
        if self.fly:
            return
        self.hit_count += 1
        if self.hit_count == 1:
            self.set_texture(1)
            self.center_x += 40
            arcade.play_sound(self.hit_sound, volume=0.5)
            self.set_texture(2)
            self.center_x += 10
            self.fly = True

       
    def update(self, delta_time: float = 1/60):
        if self.fly:
            self.fly_timer += delta_time
            if self.fly_timer >= 0.07:
                self.center_x += 7
                self.angle -= 70
                if self.scale > 0.1:
                    self.scale -= 0.03
                self.fly_timer = 0.0
                if self.center_x > SCREEN_WIDTH - 30:
                    self.kill()

# ---------------- Игровое окно ----------------
class VoiceBowlingGame(arcade.Window):
    def __init__(self, task=None):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        self.task = task or {"duration": 2, "text": "МА"}  # дефолт
        self.background_texture = None
        self.player = None
        self.ball = None
        self.pins = arcade.SpriteList()
        self.is_moving = False
        self.ball_speed = 0
        self.speech_task = None


        self.target_x = 900
        #self.move_duration = 2
        # self.total_distance = self.target_x - 110
        # self.ball_speed = self.total_distance / (self.move_duration * 60)

        self.motion_start_time = None
        self.movement_timer = 0
        self.timer_frozen = False
        self.failed_attempt = False

    def setup(self):
        arcade.set_background_color(arcade.color.LIGHT_BLUE)
        self.background_texture = arcade.load_texture("fon_bouling_4.png")
        


        self.player = arcade.Sprite("keglya/idle1.png", scale=0.25)
        self.player.center_x = 100
        self.player.center_y = 160

        self.move_duration = int(self.task.get("duration", 2))
        self.total_distance = self.target_x - 110
        self.ball_speed = self.total_distance / (self.move_duration * 60)

        font_path = str(Path("alphabet/shrift_mult.otf").absolute())
        self.task_sprite = arcade.create_text_sprite(
            text=self.task.get("text", "МА"),
            start_x=510,
            start_y=200,
            font_size=60,
            font_name=font_path,
            color=arcade.color.RED_DEVIL,
            anchor_x="center",
            anchor_y="center"
        )
        self.ball = arcade.Sprite("keglya/shar.png", scale=0.05)
        self.ball.center_x = 115
        self.ball.center_y = 95

        self.pins = arcade.SpriteList()
        self.pins.append(Keglya(900, 130))
        

        self.motion_start_time = None
        self.movement_timer = 0
        self.timer_frozen = False
        self.is_moving = False
        self.failed_attempt = False

        analyze_voice(self.on_voice_detected)

    def on_voice_detected(self, is_voiced):
        if is_voiced and not self.is_moving and not self.timer_frozen:
            self.motion_start_time = time.time()
            self.player = arcade.Sprite("keglya/idle2.png", scale=0.25)
            self.player.center_x = 100
            self.player.center_y = 160

        elif not is_voiced and self.is_moving and not self.timer_frozen:
            self.failed_attempt = True
            self.is_moving = False
        self.is_moving = bool(is_voiced)

    def on_draw(self):
        arcade.start_render()
        arcade.draw_lrwh_rectangle_textured(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, self.background_texture)

        self.player.draw()
        self.task_sprite.draw()
        
        self.ball.draw()
        self.pins.draw()
        


        if self.motion_start_time and not self.timer_frozen:
            self.movement_timer = time.time() - self.motion_start_time

        arcade.draw_text(f"Ball time: {self.movement_timer:.2f} s",
                         SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30,
                         arcade.color.BLACK, 20, anchor_x="center")

        arcade.draw_text("Press R to restart", SCREEN_WIDTH - 180, 10, arcade.color.DARK_GRAY, 14)

    def on_update(self, delta_time: float):
        if self.failed_attempt:
            self.player = arcade.Sprite("keglya/idle4.png", scale=0.25)
            self.player.center_x = 100
            self.player.center_y = 160
            if self.ball.center_y > 15:
                self.ball.center_x += 8
                self.ball.center_y -= 5
                self.ball.angle -= 10
            else:
                self.ball.center_x += 8 # только вправо, вниз больше не едет
                self.ball.angle -= 10 

            if self.ball.center_x > SCREEN_WIDTH:
                self.timer_frozen = True
                self.failed_attempt = False
            return

        if self.is_moving and self.ball.center_x < self.target_x:
            self.ball.center_x += self.ball_speed
            self.ball.angle -= 10
        elif self.is_moving and self.ball.center_x >= self.target_x:
            self.is_moving = False
            self.timer_frozen = True

        for keglya in self.pins:
            keglya.update(delta_time)

        for keglya in self.pins:
            if arcade.check_for_collision(self.ball, keglya):
                keglya.on_hit()
                self.player = arcade.Sprite("keglya/idle3.png", scale=0.25)
                self.player.center_x = 100
                self.player.center_y = 160

    def on_key_press(self, key, modifiers):
        if key == arcade.key.R:
            self.setup()

# ---------------- Запуск ----------------
if __name__ == "__main__":
    game = VoiceBowlingGame()
    game.setup()
    arcade.run()
