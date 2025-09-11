import arcade
from pathlib import Path

from . import settings as s


class Gem:
    def __init__(self, x, y, word):
        self.letters = arcade.SpriteList()
        self.change_x = -s.SCROLL_SPEED

        font_path = str(Path("games/slogotakt/components/shrift_mult.otf").absolute())
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
            letter_sprite.true_x = x + i * letter_spacing
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
        for letter_sprite in list(self.letters):
            if arcade.check_for_collision(player, letter_sprite):
                self.letters.remove(letter_sprite)
                return True
        return False
