import arcade
from . import settings as s


def load_texture_pair(filename):
    return [
        arcade.load_texture(filename),
        arcade.load_texture(filename, flipped_horizontally=True),
    ]


class PlayerCharacter(arcade.Sprite):
    def __init__(self):
        super().__init__()
        main_path = ":resources:images/animated_characters/male_person/malePerson"

        self.scale = 0.5
        self.character_face_direction = 0
        self.cur_texture = 0
        self.animation_timer = 0.0

        self.idle_texture_pair = load_texture_pair(f"{main_path}_idle.png")
        self.jump_texture_pair = load_texture_pair(f"{main_path}_jump.png")
        self.fall_texture_pair = load_texture_pair(f"{main_path}_fall.png")

        self.walk_textures = []
        for i in range(8):
            texture = load_texture_pair(f"{main_path}_walk{i}.png")
            self.walk_textures.append(texture)

        self.texture = self.idle_texture_pair[self.character_face_direction]
        self.center_x = 150
        self.center_y = s.GROUND_Y

        self.hit_box = self.texture.hit_box_points
        self.velocity_y = 0.0

    def update_physics(self, is_voiced: bool):
        if is_voiced:
            self.velocity_y += s.UP_FORCE
            if self.velocity_y > s.MAX_UP_SPEED:
                self.velocity_y = s.MAX_UP_SPEED
        else:
            self.velocity_y -= s.GRAVITY_FORCE
            if self.velocity_y < -s.MAX_DOWN_SPEED:
                self.velocity_y = -s.MAX_DOWN_SPEED

        self.center_y += self.velocity_y

        if self.center_y < s.GROUND_Y:
            self.center_y = s.GROUND_Y
            self.velocity_y = 0
        elif self.center_y > s.AIR_Y:
            self.center_y = s.AIR_Y
            self.velocity_y = 0

    def update_animation(self, delta_time: float = 1 / 60):
        self.animation_timer += delta_time

        if self.center_y <= s.GROUND_Y + 0.1:
            if abs(self.velocity_y) < 0.1:
                if self.animation_timer >= 0.1:
                    self.animation_timer = 0.0
                    self.cur_texture = (self.cur_texture + 1) % 8
                self.texture = self.walk_textures[self.cur_texture][self.character_face_direction]
            else:
                self.texture = self.walk_textures[self.cur_texture][self.character_face_direction]
        elif self.center_y >= s.AIR_Y - 0.1:
            if abs(self.velocity_y) < 0.1:
                self.texture = self.idle_texture_pair[self.character_face_direction]
            elif self.velocity_y > 0:
                self.texture = self.jump_texture_pair[self.character_face_direction]
            else:
                self.texture = self.fall_texture_pair[self.character_face_direction]
        else:
            self.texture = self.jump_texture_pair[self.character_face_direction] if self.velocity_y > 0 else self.fall_texture_pair[self.character_face_direction]
