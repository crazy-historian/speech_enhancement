import arcade
from typing import Optional

from . import settings as s


class PlayerCharacter(arcade.Sprite):
    def __init__(self, scale=0.15):
        super().__init__()
        self.textures = [arcade.load_texture(f"games/gromik/components/{i}_bird.png") for i in range(1, 7)]
        self.texture_index = 0
        self.set_texture(self.texture_index)
        
        self.scale = scale
        self.center_x = 150
        self.center_y = s.GROUND_Y
        self.target_y = s.GROUND_Y
        self.frame_timer = 0.0

    def update_animation(self, delta_time: float = 1/60):
        self.frame_timer += delta_time
        if self.frame_timer > 0.1:
            self.texture_index = (self.texture_index + 1) % len(self.textures)
            self.set_texture(self.texture_index)
            self.frame_timer = 0.0

    def update_position(self, intensity: Optional[float]) -> None:
        if intensity is None:
            return

        # поддержим оба набора настроек: *_DB и без суффикса
        min_db = getattr(s, "INTENSITY_MIN_DB", getattr(s, "INTENSITY_MIN", 40.0))
        max_db = getattr(s, "INTENSITY_MAX_DB", getattr(s, "INTENSITY_MAX", 90.0))

        span = max(1e-6, (float(max_db) - float(min_db)))
        normalized = (float(intensity) - float(min_db)) / span
        normalized = max(0.0, min(1.0, normalized))

        target_position = s.GROUND_Y + normalized * (s.AIR_Y - s.GROUND_Y)
        alpha = float(getattr(s, "SMOOTHING_ALPHA", 0.6))
        response = float(getattr(s, "RESPONSE_FACTOR", 0.9))

        # сглажённое приближение
        self.target_y = alpha * target_position + (1 - alpha) * getattr(self, "target_y", s.GROUND_Y)

        if abs(self.center_y - self.target_y) > 20:
            self.center_y += (self.target_y - self.center_y) * response
        else:
            self.center_y = self.target_y

        self.center_y = max(s.GROUND_Y, min(self.center_y, s.AIR_Y))