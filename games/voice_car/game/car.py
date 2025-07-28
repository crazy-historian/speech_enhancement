import arcade
from game.settings import PITCH_MAX, PITCH_MIN, SCREEN_HEIGHT, SMOOTHING_FACTOR, START_X, TRACK_CENTER_Y, TRACK_HEIGHT


class Car(arcade.Sprite):
    def __init__(self):
        super().__init__("games/voice_car/components/car.png", scale=0.25)
        self.center_x = START_X
        self.center_y = SCREEN_HEIGHT -350
        self.moving_forward = False
        self.out_of_zone = False
        self.rotating = False
        self.rotation_speed = 300  # градусов в секунду

        # Загружаем все текстуры
        self.frames = [
            arcade.load_texture(f"games/voice_car/components/car_{i}.png") for i in range(1, 5)
        ]
        self.current_frame = 0
        self.animation_timer = 0.0
        self.frame_duration = 0.1  # сколько секунд показывать один кадр
    def start_rotation(self):
        self.rotating = True
    
    def update_rotation(self, delta_time):
        if self.rotating:
            self.angle += self.rotation_speed * delta_time
    
    def update_animation(self, delta_time: float):
        self.animation_timer += delta_time
        if self.animation_timer >= self.frame_duration:
            self.animation_timer = 0.0
            self.current_frame = (self.current_frame + 1) % len(self.frames)
            self.texture = self.frames[self.current_frame]

    def update_position(self, voice_status, pitch, delta_time):
        """
        Двигаем корабль по логике:
          - Если есть голос и pitch в зоне (PITCH_MIN..PITCH_MAX) → двигаем по Y в рамках зелёной области
          - Если есть голос, но вышел из зоны → всё равно двигаем по X, Y растягиваем к "красным" зонам
          - Если голоса нет → корабль останавливается, Y тянется к центру
        """
        if voice_status == 1 and PITCH_MIN <= pitch <= PITCH_MAX:
            self.moving_forward = True

            pitch_range = PITCH_MAX - PITCH_MIN
            relative_position = (pitch - PITCH_MIN) / pitch_range

            # Вместо SCREEN_HEIGHT // 2 используем TRACK_CENTER_Y
            target_y = TRACK_CENTER_Y - TRACK_HEIGHT // 2 + relative_position * TRACK_HEIGHT
            self.center_y += (target_y - self.center_y) * SMOOTHING_FACTOR

        elif voice_status == 1:
            self.moving_forward = True
            direction = "up" if pitch > PITCH_MAX else "down"
            overshoot = abs(pitch - (PITCH_MAX if direction == "up" else PITCH_MIN)) / 100
            offset = overshoot * (TRACK_CENTER_Y - 100)

            if direction == "up":
                target_y = TRACK_CENTER_Y + TRACK_HEIGHT // 2 + offset
            else:
                target_y = TRACK_CENTER_Y - TRACK_HEIGHT // 2 - offset

            self.center_y += (target_y - self.center_y) * SMOOTHING_FACTOR

        else:
            self.moving_forward = False
            self.center_y += (TRACK_CENTER_Y - self.center_y) * 0.1

        # if self.moving_forward and self.center_x < END_X:
        #     self.center_x += SHIP_SPEED * delta_time
