import arcade
from pathlib import Path
from game.settings import PITCH_FLOOR, PITCH_CEILING, AIR_Y, GROUND_Y, SCREEN_WIDTH, SCROLL_SPEED


class ArtifactGroup:
    def __init__(self, task_text, pitch):
        self.letters = []
        spacing = 5  
        font_path = str(Path("games/pichik/components/shrift_mult.otf").absolute())
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