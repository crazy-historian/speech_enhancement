import arcade
from pathlib import Path
from game.settings import SCROLL_SPEED

class Gem:
    """
    Гем, состоящий из нескольких букв (sprite'ов текста на основе кастомного шрифта).
    """
    def __init__(self, x, y, word):
        self.letters = arcade.SpriteList()
        self.change_x = -SCROLL_SPEED

        font_path = str(Path("games/slogotakt/components/shrift_mult.otf").absolute())  # ваш путь к OTF
        color = (165, 42, 42, 255)
        letter_spacing = 60

        for i, letter in enumerate(word):
            letter_sprite = arcade.create_text_sprite(
                text=letter,
                start_x=x + i * letter_spacing,
                start_y=y,
                font_size=48,
                font_name=font_path,
                color=color,
                anchor_x="center",
                anchor_y="center"
            )
            letter_sprite.true_x = x + i * letter_spacing  # для корректной анимации
            letter_sprite.center_y = y
            self.letters.append(letter_sprite)

    def update(self):
        for letter_sprite in self.letters:
            letter_sprite.true_x += self.change_x
            letter_sprite.center_x = round(letter_sprite.true_x)

    def draw(self):
        self.letters.draw()

    @property
    def right(self):
        if len(self.letters) == 0:
            return 0
        return max(letter.right for letter in self.letters)

    def remove_from_sprite_lists(self):
        self.letters.clear()
    
    def check_collision_and_collect(self, player):
        for letter_sprite in self.letters:
            if arcade.check_for_collision(player, letter_sprite):
                self.letters.remove(letter_sprite)  # удаляем только собранную букву
                return True  # была коллизия
        return False
