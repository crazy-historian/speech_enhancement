import arcade

from game.settings import GROUND_Y, INTENSITY_MAX, INTENSITY_MIN, SMOOTHING_ALPHA, RESPONSE_FACTOR, AIR_Y


class PlayerCharacter(arcade.Sprite):
    def __init__(self, scale=0.15):
        super().__init__()
        self.textures = [arcade.load_texture(f"games/gromik/components/{i}_bird.png") for i in range(1, 7)]
        self.texture_index = 0
        self.set_texture(self.texture_index)
        
        self.scale = scale
        self.center_x = 150
        self.center_y = GROUND_Y
        self.target_y = GROUND_Y
        self.frame_timer = 0.0

    def update_animation(self, delta_time: float = 1/60):
        self.frame_timer += delta_time
        if self.frame_timer > 0.1:
            self.texture_index = (self.texture_index + 1) % len(self.textures)
            self.set_texture(self.texture_index)
            self.frame_timer = 0.0

    def update_position(self, intensity: float):
            
        
            normalized = (intensity - INTENSITY_MIN) / (INTENSITY_MAX - INTENSITY_MIN)
            normalized = max(0.0, min(normalized, 1.0))  # ограничение в пределах 0–1
        
            target_position = GROUND_Y + normalized * (AIR_Y - GROUND_Y)
            self.target_y = SMOOTHING_ALPHA * target_position + (1 - SMOOTHING_ALPHA) * self.target_y

            if abs(self.center_y - self.target_y) > 20:
                self.center_y += (self.target_y - self.center_y) * RESPONSE_FACTOR
            else:
                self.center_y = self.target_y

            self.center_y = max(GROUND_Y, min(self.center_y, AIR_Y))