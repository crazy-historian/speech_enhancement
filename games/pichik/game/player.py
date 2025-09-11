import arcade
from .settings import (
    GROUND_Y,           # Y-координата земли (нижняя позиция игрока)
    PITCH_FLOOR,        # Минимальное значение pitch, воспринимаемое как голос
    PITCH_CEILING,      # Максимальное значение pitch
    SMOOTHING_ALPHA,    # Коэффициент сглаживания для плавного движения
    AIR_Y,              # Верхняя граница полёта игрока
    RESPONSE_FACTOR     # Скорость приближения к целевой позиции
)


class PlayerCharacter(arcade.Sprite):
    """
    Спрайт игрока, управляемый с помощью голосового pitch (частоты голоса).

    Персонаж поднимается выше при повышении pitch и опускается при его снижении.
    Анимация крыльев обновляется автоматически.
    """

    def __init__(self, scale=0.15):
        """
        Инициализирует игрока: загружает текстуры, устанавливает начальную позицию.

        Args:
            scale (float): Масштаб спрайта (по умолчанию 0.15).
        """
        super().__init__()

        # Загружаем 6 кадров анимации из папки components
        self.textures = [
            arcade.load_texture(f"games/pichik/components/{i}_bird.png")
            for i in range(1, 7)
        ]
        self.texture_index = 0  # Начальный кадр анимации
        self.set_texture(self.texture_index)  # Устанавливаем первую текстуру

        self.scale = scale
        self.center_x = 150  # Фиксированная X-позиция (игрок не двигается по горизонтали)
        self.center_y = GROUND_Y  # Начинаем на земле
        self.target_y = GROUND_Y  # Целевая Y-позиция (для плавного движения)
        self.frame_timer = 0.0  # Таймер для переключения кадров анимации

    def update_animation(self, delta_time: float = 1/60):
        """
        Обновляет анимацию крыльев (переключает текстуры) с заданной частотой.

        Args:
            delta_time (float): Время, прошедшее с последнего кадра (обычно ~1/60 сек).
        """
        self.frame_timer += delta_time
        # Меняем кадр каждые 0.1 секунды
        if self.frame_timer > 0.1:
            self.texture_index = (self.texture_index + 1) % len(self.textures)
            self.set_texture(self.texture_index)
            self.frame_timer = 0.0  # Сбрасываем таймер

    def update_position(self, pitch: float):
        """
        Обновляет вертикальную позицию игрока на основе текущего pitch (частоты голоса).

        Если pitch отсутствует (None), игрок медленно опускается вниз.

        Args:
            pitch (float or None): Текущее значение pitch (в Гц), полученное от микрофона.
        """
        if pitch is None:
            # Если голос не слышен, плавно опускаем игрока вниз (симуляция "падения")
            pitch = 100  # Используем базовое значение ниже PITCH_FLOOR

        # Преобразуем pitch в целевую Y-позицию в диапазоне [GROUND_Y, AIR_Y]
        target_position = GROUND_Y + (
            (pitch - PITCH_FLOOR) / (PITCH_CEILING - PITCH_FLOOR)
        ) * (AIR_Y - GROUND_Y)

        # Плавное движение: усредняем новую позицию с предыдущей (сглаживание)
        self.target_y = (
            SMOOTHING_ALPHA * target_position +
            (1 - SMOOTHING_ALPHA) * self.target_y
        )

        # Плавно приближаемся к целевой позиции
        if abs(self.center_y - self.target_y) > 20:
            self.center_y += (self.target_y - self.center_y) * RESPONSE_FACTOR
        else:
            # Если близко — просто прыгаем в цель (избегаем дрожания)
            self.center_y = self.target_y

        # Ограничиваем движение в пределах экрана
        self.center_y = max(GROUND_Y, min(self.center_y, AIR_Y))