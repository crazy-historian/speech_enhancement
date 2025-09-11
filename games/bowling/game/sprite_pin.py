import arcade
from . import settings as s


class Keglya(arcade.Sprite):
    def __init__(self, x, y):
        super().__init__("games/bowling/components/keglya_1.png", scale=0.7)
        self.textures = [
            arcade.load_texture("games/bowling/components/keglya_1.png"),
            arcade.load_texture("games/bowling/components/keglya_2.png"),
            arcade.load_texture("games/bowling/components/keglya_3.png")
        ]
        self.set_texture(0)
        self.hit_sound = arcade.load_sound("games/bowling/components/hit.wav")
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
                if self.center_x > s.SCREEN_WIDTH - 30:
                    self.kill()