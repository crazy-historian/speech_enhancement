import arcade
import time
from pathlib import Path
from . import settings as s

from .voice import analyze_voice
from .sprite_pin import Keglya

# 👇 добавь импорт селектора задач (относительный путь)
from ..guui.qt_task_selector import select_task_for_profile


class VoiceBowlingGame(arcade.Window):
    def __init__(self, task=None):
        super().__init__(s.SCREEN_WIDTH, s.SCREEN_HEIGHT, s.SCREEN_TITLE)
        self.task = task or {"duration": 2, "text": "МА"}
        self.background_texture = None
        self.player = None
        self.ball = None
        self.pins = arcade.SpriteList()
        self.ball_speed = 0
        self.target_x = 900
        self.task_sprite = None

        # состояние движения
        self.reset_game_state()

        # 👇 флаги паузы/перезапуска (как в pichik)
        self.paused = False
        self.pause_start = None
        self.restart_requested = False

    def reset_game_state(self):
        self.motion_start_time = None
        self.movement_timer = 0
        self.timer_frozen = False
        self.is_moving = False
        self.failed_attempt = False

    def set_player_sprite(self, image_path: str):
        self.player = arcade.Sprite(image_path, scale=0.25)
        self.player.center_x = 100
        self.player.center_y = 160

    def setup(self):
        arcade.set_background_color(arcade.color.LIGHT_BLUE)
        self.background_texture = arcade.load_texture("games/bowling/components/fon_bouling_4.png")
        self.set_player_sprite("games/bowling/components/idle1.png")

        # 👇 если передали profile_name в task — сохраним (нужно для qt_task_selector)
        prof = self.task.get("profile_name")
        if prof:
            s.profile_name = prof

        self.move_duration = int(self.task.get("duration", 2))
        self.total_distance = self.target_x - 110
        self.ball_speed = self.total_distance / (self.move_duration * 60)

        font_path = str(Path("games/bowling/components/shrift_mult.otf").absolute())
        self.task_sprite = arcade.create_text_sprite(
            text=self.task.get("text", "МА"),
            start_x=510,
            start_y=200,
            font_size=60,
            font_name=font_path,
            color=arcade.color.RED_DEVIL,
            anchor_x="center",
            anchor_y="center"
        )

        self.ball = arcade.Sprite("games/bowling/components/shar.png", scale=0.05)
        self.ball.center_x = 115
        self.ball.center_y = 95

        self.pins = arcade.SpriteList()
        self.pins.append(Keglya(900, 130))

        self.reset_game_state()
        analyze_voice(self.on_voice_detected)

        # сбросим паузу/перезапуск
        self.paused = False
        self.pause_start = None
        self.restart_requested = False

    def on_voice_detected(self, is_voiced):
        if is_voiced and not self.is_moving and not self.timer_frozen and not self.paused:
            self.motion_start_time = time.time()
            self.set_player_sprite("games/bowling/components/idle2.png")
            self.is_moving = True
        elif (not is_voiced) and self.is_moving and not self.timer_frozen:
            self.failed_attempt = True
            self.is_moving = False

    def on_draw(self):
        arcade.start_render()
        arcade.draw_lrwh_rectangle_textured(0, 0, s.SCREEN_WIDTH, s.SCREEN_HEIGHT, self.background_texture)
        self.player.draw()
        self.task_sprite.draw()
        self.ball.draw()
        self.pins.draw()

        # 👇 во время паузы не увеличиваем таймер движения
        if self.motion_start_time and not self.timer_frozen and not self.paused:
            self.movement_timer = time.time() - self.motion_start_time

    def on_update(self, delta_time: float):
        # 👇 безопасный перезапуск по флагу
        if self.restart_requested:
            self.restart_requested = False
            self.setup()
            return

        if self.paused:
            return

        if self.failed_attempt:
            self.set_player_sprite("games/bowling/components/idle4.png")
            if self.ball.center_y > 15:
                self.ball.center_x += 8
                self.ball.center_y -= 5
                self.ball.angle -= 10
            else:
                self.ball.center_x += 8
                self.ball.angle -= 10

            if self.ball.center_x > s.SCREEN_WIDTH:
                self.timer_frozen = True
                self.failed_attempt = False
            return

        if self.is_moving and self.ball.center_x < self.target_x:
            self.ball.center_x += self.ball_speed
            self.ball.angle -= 10
        elif self.is_moving and self.ball.center_x >= self.target_x:
            self.is_moving = False
            self.timer_frozen = True

        for keglya in self.pins:
            keglya.update(delta_time)

        for keglya in self.pins:
            if arcade.check_for_collision(self.ball, keglya):
                keglya.on_hit()
                self.set_player_sprite("games/bowling/components/idle3.png")

    def on_key_press(self, key, modifiers):
        if key == arcade.key.R:
            self.setup()
        elif key == arcade.key.T:
            self.open_task_selection_dialog()

    # ---------- НОВОЕ: выбор задания на лету ----------
    def restart_with_task(self, task: dict):
        """
        Обновляем задание и просим мягкий перезапуск.
        Для боулинга достаточно duration + text (+ profile_name опционально).
        """
        # подмешаем профиль, если пришёл
        prof = task.get("profile_name")
        if prof:
            s.profile_name = prof

        # обновим активный task
        self.task = {
            **self.task,
            "duration": int(task.get("duration", self.task.get("duration", 2))),
            "text": task.get("text", self.task.get("text", "МА")),
            "profile_name": s.profile_name,
        }

        self.restart_requested = True
        self.paused = False
        self.pause_start = None

    def open_task_selection_dialog(self):
        """
        Ставит игру на паузу, открывает Qt-диалог выбора задания,
        компенсирует время паузы в таймере движения и перезапускает игру с новым task.
        """
        was_moving = bool(self.is_moving and self.motion_start_time and not self.timer_frozen)

        # ставим паузу
        self.paused = True
        self.pause_start = time.time()

        # показываем диалог (внутри — ручной processEvents)
        profile = getattr(s, "profile_name", "") or ""
        try:
            task = select_task_for_profile(profile)
        except Exception as e:
            print("[TaskPicker] exception:", e)
            task = None

        # снимаем паузу
        paused_dur = time.time() - self.pause_start if self.pause_start else 0
        self.pause_start = None
        self.paused = False

        # если во время паузы шло движение — компенсируем таймер
        if was_moving and self.motion_start_time and not self.timer_frozen:
            self.motion_start_time += paused_dur  # чтобы movement_timer не скакал

        if task:
            self.restart_with_task(task)
