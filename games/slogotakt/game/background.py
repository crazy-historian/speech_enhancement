import arcade
from . import settings as s


class ScrollingBackground:
    def __init__(self, texture_path, speed):
        self.speed = speed
        self.width = s.SCREEN_WIDTH
        self.height = s.SCREEN_HEIGHT

        self.textures = [
            arcade.Sprite(texture_path, scale=1.0),
            arcade.Sprite(texture_path, scale=1.0),
        ]

        for sprite in self.textures:
            sprite.width = self.width
            sprite.height = self.height

        self.textures[0].center_x = s.SCREEN_WIDTH // 2
        self.textures[0].center_y = s.SCREEN_HEIGHT // 2

        self.textures[1].center_x = s.SCREEN_WIDTH + s.SCREEN_WIDTH // 2
        self.textures[1].center_y = s.SCREEN_HEIGHT // 2

    def update(self):
        for sprite in self.textures:
            sprite.center_x -= self.speed
            if sprite.right < 0:
                sprite.center_x += s.SCREEN_WIDTH * 2

    def draw(self):
        for sprite in self.textures:
            sprite.draw()
