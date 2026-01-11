# core_game.py
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Dict, Any
from collections import deque

import arcade
from arcade.types import LBWH

from . import settings as s
from .voice import analyze_voice
from .sprite_pin import Keglya
from ..guui.qt_task_selector import select_task_for_profile


class VoiceBowlingGame(arcade.Window):
    def __init__(self, task: Optional[Dict[str, Any]] = None):
        super().__init__(s.SCREEN_WIDTH, s.SCREEN_HEIGHT, s.SCREEN_TITLE)

        self.task: Dict[str, Any] = task or {"duration": 2, "text": "МА"}

        # --- графика ---
        self.background_texture: Optional[arcade.Texture] = None

        self.player: Optional[arcade.Sprite] = None
        self.player_textures: list[arcade.Texture] = []

        self.ball: Optional[arcade.Sprite] = None

        self.actors: arcade.SpriteList = arcade.SpriteList()
        self.pins: arcade.SpriteList = arcade.SpriteList()

        self.task_text: Optional[arcade.Text] = None

        # --- движение ---
        self.ball_speed: float = 0.0
        self.target_x: float = 900.0
        self.move_duration: int = 2
        self.total_distance: float = 0.0

        # --- состояние ---
        self.reset_game_state()
        self.paused: bool = False
        self.pause_start: Optional[float] = None
        self.restart_requested: bool = False

        # --- голос (потокобезопасно) ---
        self._voice_events: deque[bool] = deque()

    def reset_game_state(self) -> None:
        self.motion_start_time: Optional[float] = None
        self.movement_timer: float = 0.0
        self.timer_frozen: bool = False
        self.is_moving: bool = False
        self.failed_attempt: bool = False

    # ---------- игрок ----------
    def _load_player_textures(self) -> None:
        self.player_textures = [
            arcade.load_texture("games/bowling/components/idle1.png"),
            arcade.load_texture("games/bowling/components/idle2.png"),
            arcade.load_texture("games/bowling/components/idle3.png"),
            arcade.load_texture("games/bowling/components/idle4.png"),
        ]

    def set_player_state(self, idx: int) -> None:
        if self.player and 0 <= idx < len(self.player_textures):
            self.player.texture = self.player_textures[idx]

    # ---------- setup ----------
    def setup(self) -> None:
        arcade.set_background_color(arcade.color.LIGHT_BLUE)

        prof = self.task.get("profile_name")
        if prof:
            s.profile_name = prof

        self.move_duration = int(self.task.get("duration", 2))
        self.total_distance = self.target_x - 110
        self.ball_speed = self.total_distance / (self.move_duration * 60)

        self.background_texture = arcade.load_texture(
            "games/bowling/components/fon_bouling_4.png"
        )

        self.actors = arcade.SpriteList()
        self.pins = arcade.SpriteList()

        self._load_player_textures()
        self.player = arcade.Sprite(scale=0.25)
        self.player.center_x = 100
        self.player.center_y = 160
        self.player.texture = self.player_textures[0]
        self.actors.append(self.player)

        self.ball = arcade.Sprite("games/bowling/components/shar.png", scale=0.05)
        self.ball.center_x = 115
        self.ball.center_y = 95
        self.actors.append(self.ball)

        self.pins.append(Keglya(900, 130))

        # ---------- font (robust) ----------
        # core_game.py: games/bowling/game/core_game.py
        # components:  games/bowling/components/shrift_mult.otf
        font_path = (Path(__file__).resolve().parents[1] / "components" / "shrift_mult.otf")

        try:
            arcade.load_font(str(font_path))
            # IMPORTANT: s.FONT_FAMILY должен быть реальным "family name" шрифта
            font_name = getattr(s, "FONT_FAMILY", None)
        except Exception as e:
            print("[Font] load failed:", e, "path:", font_path)
            font_name = None

        self.task_text = arcade.Text(
            self.task.get("text", "МА"),
            510,
            200,
            arcade.color.RED_DEVIL,
            60,
            anchor_x="center",
            anchor_y="center",
            font_name=font_name,
        )

        self.reset_game_state()
        self._voice_events.clear()

        # ВАЖНО: callback теперь безопасный
        analyze_voice(self.on_voice_detected)

        self.paused = False
        self.pause_start = None
        self.restart_requested = False

    # ---------- голос (из другого потока!) ----------
    def on_voice_detected(self, is_voiced: bool) -> None:
        # НИЧЕГО не трогаем из arcade здесь
        self._voice_events.append(bool(is_voiced))

    # ---------- draw ----------
    def _draw_background(self) -> None:
        if self.background_texture:
            arcade.draw_texture_rect(
                self.background_texture,
                LBWH(0, 0, self.width, self.height),
            )

    def on_draw(self) -> None:
        self.clear()
        self._draw_background()
        self.actors.draw()
        self.pins.draw()
        if self.task_text:
            self.task_text.draw()

    # ---------- update ----------
    def on_update(self, delta_time: float) -> None:
        if self.restart_requested:
            self.restart_requested = False
            self.setup()
            return

        if self.paused:
            return

        # --- обработка событий голоса (ГЛАВНЫЙ ПОТОК) ---
        while self._voice_events:
            is_voiced = self._voice_events.popleft()

            if is_voiced and not self.is_moving and not self.timer_frozen:
                self.motion_start_time = time.time()
                self.set_player_state(1)
                self.is_moving = True

            elif not is_voiced and self.is_moving and not self.timer_frozen:
                self.failed_attempt = True
                self.is_moving = False

        if self.failed_attempt and self.ball:
            self.set_player_state(3)
            if self.ball.center_y > 15:
                self.ball.center_x += 8
                self.ball.center_y -= 5
            else:
                self.ball.center_x += 8
            self.ball.angle -= 10

            if self.ball.center_x > s.SCREEN_WIDTH:
                self.timer_frozen = True
                self.failed_attempt = False
            return

        if self.is_moving and self.ball:
            if self.ball.center_x < self.target_x:
                self.ball.center_x += self.ball_speed
                self.ball.angle -= 10
            else:
                self.is_moving = False
                self.timer_frozen = True

        for keglya in self.pins:
            keglya.update(delta_time)

        if self.ball:
            for keglya in self.pins:
                if arcade.check_for_collision(self.ball, keglya):
                    keglya.on_hit()
                    self.set_player_state(2)

    # ---------- input ----------
    def on_key_press(self, key: int, modifiers: int) -> None:
        if key == arcade.key.R:
            self.setup()
        elif key == arcade.key.T:
            self.open_task_selection_dialog()

    # ---------- задачи ----------
    def restart_with_task(self, task: Dict[str, Any]) -> None:
        prof = task.get("profile_name")
        if prof:
            s.profile_name = prof

        self.task = {
            **self.task,
            "duration": int(task.get("duration", self.task.get("duration", 2))),
            "text": task.get("text", self.task.get("text", "МА")),
            "profile_name": s.profile_name,
        }

        if self.task_text:
            self.task_text.text = self.task["text"]

        self.restart_requested = True
        self.paused = False
        self.pause_start = None

    def open_task_selection_dialog(self) -> None:
        was_moving = bool(self.is_moving and self.motion_start_time and not self.timer_frozen)

        self.paused = True
        self.pause_start = time.time()

        profile = s.profile_name or ""
        try:
            task = select_task_for_profile(profile)
        except Exception as e:
            print("[TaskPicker] exception:", e)
            task = None

        paused_dur = time.time() - self.pause_start
        self.pause_start = None
        self.paused = False

        if was_moving and self.motion_start_time and not self.timer_frozen:
            self.motion_start_time += paused_dur

        if task:
            self.restart_with_task(task)
