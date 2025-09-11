# games/voice_car/game/car.py
import math
import arcade
from . import settings as s


class Car(arcade.Sprite):
    def __init__(self):
        super().__init__("games/voice_car/components/car.png", scale=0.25)
        self.center_x = s.START_X
        self.center_y = s.SCREEN_HEIGHT - 350
        self.moving_forward = False
        self.out_of_zone = False
        self.rotating = False
        self.rotation_speed = 300  # градусов в секунду

        # Кадры анимации
        self.frames = [
            arcade.load_texture(f"games/voice_car/components/car_{i}.png") for i in range(1, 5)
        ]
        self.current_frame = 0
        self.animation_timer = 0.0
        self.frame_duration = 0.1
        # Установим стартовую текстуру
        if self.frames:
            self.texture = self.frames[0]

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

    def _is_valid_pitch(self, value) -> bool:
        return isinstance(value, (int, float)) and not math.isnan(value)

    def update_position(self, voice_status, pitch, delta_time):
        """
        Логика:
          - Голос есть и pitch в зоне → движемся внутри зелёной дорожки по Y
          - Голос есть, но вне зоны → тянем за пределы дорожки (к "красным" зонам)
          - Голоса нет или pitch невалиден → не едем, тянем к центру
        """
        # Нормализуем pitch
        p = float(pitch) if self._is_valid_pitch(pitch) else None

        # Нет голоса или нет валидного питча — безопасный режим
        if voice_status != 1 or p is None:
            self.moving_forward = False
            self.center_y += (s.TRACK_CENTER_Y - self.center_y) * 0.1
            # Ограничим позицию по экрану
            self.center_y = max(0, min(self.center_y, s.SCREEN_HEIGHT))
            return

        # Голос есть и pitch валиден
        self.moving_forward = True

        # Внутри зоны
        if s.PITCH_MIN <= p <= s.PITCH_MAX:
            pitch_range = max(1.0, (s.PITCH_MAX - s.PITCH_MIN))  # защита от деления на 0
            relative = (p - s.PITCH_MIN) / pitch_range
            target_y = s.TRACK_CENTER_Y - s.TRACK_HEIGHT // 2 + relative * s.TRACK_HEIGHT
            self.center_y += (target_y - self.center_y) * s.SMOOTHING_FACTOR

        # Вне зоны
        else:
            # Нормируем "перелёт" относительно ширины зоны, чтобы не улетать в бесконечность
            denom = max(1.0, (s.PITCH_MAX - s.PITCH_MIN))
            if p > s.PITCH_MAX:
                overshoot = (p - s.PITCH_MAX) / denom
                target_y = s.TRACK_CENTER_Y + s.TRACK_HEIGHT // 2 + overshoot * (s.SCREEN_HEIGHT * 0.3)
            else:
                overshoot = (s.PITCH_MIN - p) / denom
                target_y = s.TRACK_CENTER_Y - s.TRACK_HEIGHT // 2 - overshoot * (s.SCREEN_HEIGHT * 0.3)
            self.center_y += (target_y - self.center_y) * s.SMOOTHING_FACTOR

        # Финальный клип по экрану
        self.center_y = max(0, min(self.center_y, s.SCREEN_HEIGHT))

        # Перемещение по X у тебя отключено — оставляю как есть
        # if self.moving_forward and self.center_x < s.END_X:
        #     self.center_x += s.SHIP_SPEED * delta_time
