from . import settings as s
import arcade 
import time
import random
import numpy as np
from .background import ScrollingBackground
from .player import PlayerCharacter
import threading
from .voice import analyze_voice
from .artifact import ArtifactGroup


class VoiceArcadeGame(arcade.Window):
    """
    Аркадная игра с голосовым управлением.

    Управляет персонажем, порождает артефакты и обрабатывает голосовые команды игрока.
    """
    def __init__(self):
        """
        Инициализирует окно игры, фон, игрока, артефакты и параметры игры.
        """
        super().__init__(s.SCREEN_WIDTH, s.SCREEN_HEIGHT, s.SCREEN_TITLE)
        self.player = None
        self.artifact_groups = []
        self.artifact_list = arcade.SpriteList()
        self.voice_thread_running = True
        self.collision_counter = 0 #тоже видимо нигде не увеличивается
        self.score = 0
        self.start_time = None
        self.paused=False
        self.restart_requested=False
        self.game_over = False
        self.current_pitch = None
        self._task_overlay = None
        
        self.frequency_artifacts = s.chastota
        self.game_duration=s.GAME_DURATION

        self.bg = ScrollingBackground("games/pichik/components/fon_bird3.png", speed=5)
        self.rating = 0 #посчитать рейтинг: кол-во очков
        self.final_text = "" #что выводится в конце игры
        self.kombo_sprite = arcade.Sprite("games/pichik/components/kombo.png", scale=0.2) #спрайт для вывода комбо
        self.kombo_sprite.center_x = s.SCREEN_WIDTH // 2 #спрайт для вывода комбо
        self.kombo_sprite.center_y = s.SCREEN_HEIGHT // 2 - 200 #спрайт для вывода комбо

        # Для волн
        self.in_wave = False
        self.artifacts_spawned = 0
        self.artifacts_in_wave = s.ARTIFACTS_IN_WAVE
        self.artifact_interval = s.ARTIFACT_INTERVAL
        self.time_since_last_artifact = 0
        self.wave_timer = 0
        self.total_artifacts = 0
        self.intensity_wave = 0  # здесь будем хранить "требуемую" чатстоту волны
        self.pitch_history = [] 


    def setup(self):
        """
        Полный сброс состояния игры + загрузка параметров из settings.
        """
        # --- базовые переменные ---
        self.player = PlayerCharacter()
        self.start_time = time.time()

        # Считываем актуальные параметры из settings (могли поменяться через выбор задания)
        self.frequency_artifacts = s.chastota
        self.game_duration = s.GAME_DURATION
        self.artifacts_in_wave = s.ARTIFACTS_IN_WAVE
        self.artifact_interval = s.ARTIFACT_INTERVAL

        # Счёт/состояния — ОБНУЛЯЕМ
        self.score = 0
        self.collision_counter = 0
        self.game_over = False
        self.paused = False
        self.restart_requested = False

        # Таймеры паузы
        self.paused_time = 0
        self.pause_start = None

        # Волны — ОБНУЛЯЕМ
        self.in_wave = False
        self.artifacts_spawned = 0
        self.time_since_last_artifact = 0
        self.total_artifacts = 0
        self.intensity_wave = 0

        # История питча — чистим
        self.pitch_history = []

        # Списки объектов — пересоздаём
        self.artifact_groups = []
        self.artifact_list = arcade.SpriteList()  # важно: новый список!

        # Чтобы первая волна могла стартовать сразу по новому chastota
        self.wave_timer = time.time() - self.frequency_artifacts

        # Перезапускаем поток голоса
        self.voice_thread_running = True
        self.voice_generator = analyze_voice()
        threading.Thread(target=self.voice_loop, daemon=True).start()
       

    def voice_loop(self):
        """ 
        Получает значения частоты основного тона, пока поток не прекратился 
        """

        for pitch in self.voice_generator:
            if not self.voice_thread_running:
                break
            self.current_pitch = pitch
            self.pitch_history.append(pitch if pitch is not None else np.nan)


    def spawn_artifacts(self):
        """
        Спавнит артефакты, пока не вышло время игры
        """

        now = time.time()
        if now - self.start_time > self.game_duration:
            return

        if not self.in_wave and (now - self.wave_timer >= self.frequency_artifacts):
            self.in_wave = True
            self.artifacts_spawned = 0
            self.time_since_last_artifact = now

        if self.in_wave:
            if (now - self.time_since_last_artifact >= self.artifact_interval
                and self.artifacts_spawned < self.artifacts_in_wave):

                # Выбираем случайный диапазон из selected_ranges
                if self.artifacts_spawned == 0:
                    intensity_range = random.choice(s.selected_ranges)
                    self.intensity_wave = random.randint(*intensity_range)

                group = ArtifactGroup(s.CURRENT_TASK_TEXT, self.intensity_wave)
                group.add_to_list(self.artifact_list)
                self.artifact_groups.append(group)

                self.artifacts_spawned += 1
                self.total_artifacts += 1
                self.time_since_last_artifact = now

            if self.artifacts_spawned >= self.artifacts_in_wave:
                self.in_wave = False
                self.wave_timer = now
    
    def on_key_press(self, symbol: int, modifiers: int):
        if symbol == arcade.key.ESCAPE:
            self.paused = not self.paused
            if self.paused:
                self.pause_start = time.time()  # запомнили, когда встали на паузу
            else:
                # При выходе с паузы добавляем длительность в paused_time
                self.paused_time += time.time() - self.pause_start
                self.pause_start = None

        elif symbol == arcade.key.R:
            self.restart_requested = True
        elif symbol == arcade.key.T:
            self.open_task_selection_dialog()
            

    
    def on_draw(self):
        """
        Отрисовка состояний игровых элементов
        """

        arcade.start_render()
        self.bg.draw()
        for group in self.artifact_groups:
            for letter in group.letters:
                letter.draw()
        self.player.draw()

        self.draw_pitch_scale()

        arcade.draw_text(
            f"Score: {self.score}",
            10, s.SCREEN_HEIGHT - 30,
            arcade.color.BLACK,
            20
        )

        if self.game_over:
            stars_path = f"games/pichik/components/star_{self.rating}.png"
            texture = arcade.load_texture(stars_path)
            arcade.draw_texture_rectangle(
                s.SCREEN_WIDTH // 2,
                s.SCREEN_HEIGHT // 2 - 50,
                texture.width,
                texture.height,
                texture
            )
            #self.kombo_sprite.draw()
        if self._task_overlay:
            self._task_overlay.draw()


    def on_update(self, delta_time):
        """
        Обновляет состояние игры каждый кадр.

        Если время не вышло, то обновляется фон, обновляется анимация персонажа,обновляется позиция персонажа, обновляется лист с артефактами, спавнятся артефакты.

        Args:
            delta_time (float): Время, прошедшее с предыдущего кадра.
        """
        if self.restart_requested:  # 🔹 теперь это выполняется первым делом
            self.voice_thread_running = False
            self.restart_requested = False
            self.game_over = False 
            self.paused=False
            self.setup()
            return

        if self.game_over:
            return
        if self.paused:
            return
        if self.restart_requested:
            self.voice_thread_running = False
            self.restart_requested = False
            self.setup()
            return

        elapsed = time.time() - self.start_time - self.paused_time
        if elapsed >= s.GAME_DURATION:
            self.game_over = True
            self.voice_thread_running = False

            max_score = self.total_artifacts * s.ARTIFACT_SCORE
            self.rating = round((self.score / max_score) * 5) if max_score > 0 else 0
            self.rating = max(1, min(self.rating, 5))
            self.final_text = f"Игра окончена!\nВы набрали {self.score} из {max_score} очков."
            return

        self.bg.update()
        self.player.update_animation(delta_time)
        self.player.update_position(self.current_pitch)

        self.artifact_list.update()
        self.spawn_artifacts()

        # Проверка коллизий
        for group in self.artifact_groups:
            if group.update_and_check(self.player, delta_time):
                self.score += s.ARTIFACT_SCORE


    def draw_pitch_scale(self):
        """
        Создает шкалу питча справа на экранае, необходимо для отладки
        """

        scale_x = s.SCREEN_WIDTH - 50
        num_steps = 5
        min_pitch = s.PITCH_FLOOR
        max_pitch = s.PITCH_CEILING

        arcade.draw_line(scale_x, s.GROUND_Y, scale_x, s.AIR_Y, arcade.color.BLACK, 2)
        for i in range(num_steps + 1):
            pitch_val = int(min_pitch + i*(max_pitch - min_pitch)/num_steps)
            y_pos = s.GROUND_Y + (i/num_steps)*(s.AIR_Y - s.GROUND_Y)
            arcade.draw_line(scale_x - 10, y_pos, scale_x+10, y_pos, arcade.color.BLACK, 2)
            arcade.draw_text(f"{pitch_val} Hz", scale_x + 15, y_pos - 10, arcade.color.BLACK, 14)

        if self.current_pitch is not None:
            pitch_y = s.GROUND_Y + ((self.current_pitch - min_pitch)/(max_pitch - min_pitch))*(s.AIR_Y - s.GROUND_Y)
            pitch_y = max(s.GROUND_Y, min(pitch_y, s.AIR_Y))
            arcade.draw_line(scale_x - 20, pitch_y, scale_x + 20, pitch_y, arcade.color.RED, 4)
    def restart_with_task(self, task: dict):
        """
        Применяем параметры таска к settings и просим перезапуск.
        """
        from . import settings as s

        # duration / частоты / текст
        s.GAME_DURATION = int(task.get("duration", s.GAME_DURATION))
        s.ARTIFACTS_IN_WAVE = int(task.get("artifacts_count", s.ARTIFACTS_IN_WAVE))
        s.ARTIFACT_INTERVAL = float(task.get("artifact_interval", s.ARTIFACT_INTERVAL))
        s.chastota = float(task.get("frequency", getattr(s, "chastota", 6)))
        s.CURRENT_TASK_TEXT = task.get("text", getattr(s, "CURRENT_TASK_TEXT", "ДА"))

        # выбранные уровни
        ranges = []
        if task.get("gen_quiet"): ranges.append((task["quiet"], task["quiet"]))
        if task.get("gen_norm"):  ranges.append((task["norm"],  task["norm"]))
        if task.get("gen_loud"):  ranges.append((task["loud"],  task["loud"]))
        if not ranges: ranges = [(100, 120)]
        s.selected_ranges = ranges

        # сглаживание
        if task.get("smooth", True):
            s.SMOOTHING_ALPHA = 0.6
            s.RESPONSE_FACTOR = 0.9
        else:
            s.SMOOTHING_ALPHA = 0.25
            s.RESPONSE_FACTOR = 0.7

        # mic_device_index остаётся в s.mic_device_index как был
        self.restart_requested = True
        self.paused = False
    

    def open_task_selection_dialog(self):
        """
        Ставит игру на паузу, открывает Qt-диалог выбора задания,
        учитывает паузу и при выборе перезапускает игру.
        """
        import time as _t
        from ..guui.qt_task_selector import select_task_for_profile
        from . import settings as s

        print("[TaskPicker] open requested")
        was_paused = bool(getattr(self, "paused", False))
        if not was_paused:
            self.paused = True
            if getattr(self, "pause_start", None) is None:
                self.pause_start = _t.time()

        try:
            task = select_task_for_profile(getattr(s, "profile_name", ""))
            print("[TaskPicker] result:", task.get("name") if task else None)
        except Exception as e:
            print("[TaskPicker] exception:", e)
            task = None

        # учесть время паузы, если мы её ставили
        if not was_paused and getattr(self, "pause_start", None) is not None:
            if hasattr(self, "paused_time"):
                self.paused_time += _t.time() - self.pause_start
            self.pause_start = None
            self.paused = False

        if task:
            if hasattr(self, "restart_with_task") and callable(self.restart_with_task):
                self.restart_with_task(task)
            else:
                self.next_task_payload = task
                self.restart_requested = True
                self.paused = False