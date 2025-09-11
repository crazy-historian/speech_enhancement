import arcade
from . import settings as s


class ScrollingBackground:
    def __init__(self, texture_path, speed):
        self.speed = speed
        texture = arcade.load_texture(texture_path)
        self.width = s.SCREEN_WIDTH
        self.height = s.SCREEN_HEIGHT

        self.textures = [
            arcade.Sprite(texture_path, scale=1.0),
            arcade.Sprite(texture_path, scale=1.0)
        ]

        for sprite in self.textures:
            sprite.width = self.width
            sprite.height = self.height

        self.textures[0].center_x = s.SCREEN_WIDTH // 2
        self.textures[0].center_y = s.SCREEN_HEIGHT // 2

        self.textures[1].center_x = s.SCREEN_WIDTH + s.SCREEN_WIDTH // 2
        self.textures[1].center_y = s.SCREEN_HEIGHT // 2

        # Добавим внутренние координаты для интерполяции
        self.true_x = [float(sprite.center_x) for sprite in self.textures]

    def update(self, delta_time):
        """
        Сдвигает поочередно спрайты фона влево, создавая бесконечный скроллирующийся фон
        """

        for i, sprite in enumerate(self.textures):
            self.true_x[i] -= self.speed * delta_time
            sprite.center_x = round(self.true_x[i])
            if sprite.right < 0:
                self.true_x[i] += s.SCREEN_WIDTH * 2
                sprite.center_x = round(self.true_x[i])

    def draw(self):
        """
        Поочередно отрисовывает спрайты фона
        """

        for sprite in self.textures:
            sprite.draw()