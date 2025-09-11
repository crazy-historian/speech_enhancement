import arcade
from pathlib import Path
from . import settings as s


class ArtifactGroup:
    def __init__(self, task, intensity):
        self.letters = []
        spacing = 5  # расстояние между буквами
        font_path = str(Path("games/gromik/components/shrift_mult.otf").absolute())
        color=(165, 42, 42, 255)  # мягкий чёрный

        normalized = (intensity - s.INTENSITY_MIN) / (s.INTENSITY_MAX - s.INTENSITY_MIN)
        normalized = max(0.0, min(normalized, 1.0))
        center_y = s.GROUND_Y + normalized * (s.AIR_Y - s.GROUND_Y)

        current_x = s.SCREEN_WIDTH + 100

        for letter in task:
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
            current_x += sprite.width + spacing  # расстояние зависит от ширины


        self.collected = False

    def add_to_list(self, sprite_list):
        """
        Функция добавляет буквы в лист артефактов группы (одна группа = одна итерация слогов для произнесения) задания
        """

        for letter in self.letters:
            sprite_list.append(letter)

    def update_and_check(self, player, delta_time):
        """
        Функция обновляет состояние группы артефактов и проверяет, собрал ли персонаж артефакт (буквы)
        """

        if self.collected:
            return False
        all_collected = True
        for sprite in self.letters:
            if sprite.collected:
                continue

            sprite.true_x -= s.SCROLL_SPEED * delta_time
            sprite.center_x = round(sprite.true_x)

            dx = abs(sprite.center_x - player.center_x)
            dy = abs(sprite.center_y - player.center_y)
            if dx <= sprite.width // 2+20 and dy <= sprite.height // 2+20:
                sprite.collected = True
                sprite.remove_from_sprite_lists()
            else:
                all_collected = False

        if all_collected:
            self.collected = True
            return True
        return False