import arcade
import time
from pathlib import Path
from game.settings import SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE
from game.voice import analyze_voice
from game.sprite_pin import Keglya


class VoiceBowlingGame(arcade.Window):
    def __init__(self, task=None):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        self.task = task or {"duration": 2, "text": "МА"}
        self.background_texture = None
        self.player = None
        self.ball = None
        self.pins = arcade.SpriteList()
        self.ball_speed = 0
        self.target_x = 900
        self.task_sprite = None

        # состояние движения
        self.reset_game_state()

    def reset_game_state(self):
        self.motion_start_time = None
        self.movement_timer = 0
        self.timer_frozen = False
        self.is_moving = False
        self.failed_attempt = False

    def set_player_sprite(self, image_path: str):
        self.player = arcade.Sprite(image_path, scale=0.25)
        self.player.center_x = 100
        self.player.center_y = 160

    def setup(self):
        arcade.set_background_color(arcade.color.LIGHT_BLUE)
        self.background_texture = arcade.load_texture("modules/components/fon_bouling_4.png")
        self.set_player_sprite("modules/components/idle1.png")

        self.move_duration = int(self.task.get("duration", 2))
        self.total_distance = self.target_x - 110
        self.ball_speed = self.total_distance / (self.move_duration * 60)

        font_path = str(Path("modules/components/shrift_mult.otf").absolute())
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

        self.ball = arcade.Sprite("modules/components/shar.png", scale=0.05)
        self.ball.center_x = 115
        self.ball.center_y = 95

        self.pins = arcade.SpriteList()
        self.pins.append(Keglya(900, 130))

        self.reset_game_state()
        analyze_voice(self.on_voice_detected)

    def on_voice_detected(self, is_voiced):
        if is_voiced and not self.is_moving and not self.timer_frozen:
            self.motion_start_time = time.time()
            self.set_player_sprite("modules/components/idle2.png")
            self.is_moving = True
        elif not is_voiced and self.is_moving and not self.timer_frozen:
            self.failed_attempt = True
            self.is_moving = False

    def on_draw(self):
        arcade.start_render()
        arcade.draw_lrwh_rectangle_textured(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, self.background_texture)
        self.player.draw()
        self.task_sprite.draw()
        self.ball.draw()
        self.pins.draw()

        if self.motion_start_time and not self.timer_frozen:
            self.movement_timer = time.time() - self.motion_start_time

        # Отладка (можно включить, если нужно)
        # arcade.draw_text(f"Ball time: {self.movement_timer:.2f} s",
        #                  SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30,
        #                  arcade.color.BLACK, 20, anchor_x="center")

    def on_update(self, delta_time: float):
        if self.failed_attempt:
            self.set_player_sprite("modules/components/idle4.png")
            if self.ball.center_y > 15:
                self.ball.center_x += 8
                self.ball.center_y -= 5
                self.ball.angle -= 10
            else:
                self.ball.center_x += 8
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
                self.set_player_sprite("modules/components/idle3.png")

    def on_key_press(self, key, modifiers):
        if key == arcade.key.R:
            self.setup()
