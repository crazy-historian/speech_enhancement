import time
import arcade
from . import settings as s

class ScrollingBackground:
    """
    Отвечает за 3 текстуры фона:
    1) bg_1 — стартовая (state='start')
    2) bg_2 — основная (state='main')
    3) bg_3 — используется в "челлендже", когда запущено задание (state='challenge'),
       затем вернёмся к bg_2.

    Параметры:
      - textures: [bg_1, bg_2, bg_3]
      - scroll_speed: скорость скролла пикселей в секунду
    """
    def __init__(self, textures, scroll_speed):
        self.textures = textures  # [bg_1, bg_2, bg_3]
        self.scroll_speed = scroll_speed
        self.active = []  # список словарей {"texture": ..., "x": ...}
        self.state = "start"
        self.challenge_active = False
        self.challenge_end_time = None
        self.reset()

    def reset(self):
        self.active = [{"texture": self.textures[0], "x": 0}]
        self.state = "start"
        self.challenge_active = False
        self.challenge_end_time = None

    def start_challenge(self, duration):
        """
        Запускает "задание" (челлендж) на указанное время.
        Текстура bg_3 добавляется так, чтобы её правый край был в x=100
        к моменту окончания задания.
        """
        self.state = "challenge"
        self.challenge_active = True
        self.challenge_end_time = time.time() + duration

            # Куда должен попасть правый край bg_3
        desired_right_x = 100
        required_offset = self.scroll_speed * duration
        start_x_bg3 = desired_right_x + required_offset - s.SCREEN_WIDTH

        # Где сейчас заканчивается последний фон
        right_edge = max(t["x"] + s.SCREEN_WIDTH for t in self.active)

        # 🔧 Добавим bg_2 до bg_3, если есть зазор
        while right_edge < start_x_bg3:
            self.active.append({
                "texture": self.textures[1],
                "x": right_edge
            })
            right_edge += s.SCREEN_WIDTH

        # ✅ Вставляем bg_3
        self.active.append({
            "texture": self.textures[2],
            "x": start_x_bg3
        })

        # ✅ И сразу за ним — bg_2, чтобы не было дыры после челленджа
        self.active.append({
            "texture": self.textures[1],
            "x": start_x_bg3 + s.SCREEN_WIDTH
        })

    def update(self, delta_time, voice_active, pitch_in_range):
        """
        Обновляет позиции текстур, если есть голос (voice_active).
        - Если state=='start', при первом голосе + pitch_in_range переходим к bg_2
        - Если challenge_active, добавляем bg_3
        - Иначе крутим bg_2 бесшовно
        """
        # Если нет голоса — фон "застывает"
        if not voice_active:
            return

        scroll_amount = self.scroll_speed * delta_time
        for tex in self.active:
            tex["x"] -= scroll_amount

        # Удаляем текстуры, которые ушли целиком за левую границу экрана
        self.active = [t for t in self.active if t["x"] + s.SCREEN_WIDTH > 0]

        # ---------------------------
        # --- Основной скроллинг
        # ---------------------------
        # Если не челлендж и не старт, крутим bg_2 циклично
        if not self.challenge_active and self.state != "start":
            # Если текстур меньше 2 (не хватает на покрытие экрана), добавим новую bg_2
            if len(self.active) < 2:
                right_edge = max(t["x"] + s.SCREEN_WIDTH for t in self.active)
                self.active.append({"texture": self.textures[1], "x": right_edge})

        # ---------------------------
        # --- Завершение челленджа
        # ---------------------------
        if self.challenge_active and time.time() >= self.challenge_end_time:
            self.challenge_active = False
            self.state = "main"
            right_edge = max(t["x"] + s.SCREEN_WIDTH for t in self.active)
            self.active.append({"texture": self.textures[1], "x": right_edge})

        # ---------------------------
        # --- Переход со старта на основной
        # ---------------------------
        if self.state == "start" and voice_active and pitch_in_range:
            self.state = "main"
            last_x = self.active[-1]["x"]
            self.active.append({"texture": self.textures[1], "x": last_x + s.SCREEN_WIDTH})

    def draw(self):
        # Рисуем каждую "полоску" фона так, чтобы её центр был по центру экрана по Y
        for tex in self.active:
            arcade.draw_texture_rectangle(
                tex["x"] + s.SCREEN_WIDTH // 2,
                s.SCREEN_HEIGHT // 2,
                s.SCREEN_WIDTH,
                s.SCREEN_HEIGHT,
                tex["texture"]
            )
