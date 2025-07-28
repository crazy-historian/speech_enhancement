import arcade
import time
import threading
from game.settings import SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE, GROUND_Y, WAVE_INTERVAL, AIR_Y
from game.background import ScrollingBackground
from game.player import PlayerCharacter
from game.voice import analyze_voice
from game.artifact import Gem


# ------------------------- Класс игры (VoiceArcadeGame) -------------------------
class VoiceArcadeGame(arcade.Window):
    def __init__(self, task, silence_threshold_db):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)

        self.silence_threshold_db = silence_threshold_db
        self.game_duration = task["duration"]
        self.wave_interval = task["wave_interval"]
        self.wave_size = task["wave_size"]
        self.syllable_interval = task["syllable_interval"]
        self.current_task = task["text"]
        self.blocks_to_silent = 0


        
        self.player = None

        self.gem_list = []
        self.gem_count = 0
        

        self.wave_in_progress = False
        
        self.wave_collected = 0
        self.wave_failed = False
        self.score = 0
        self.max_possible_score = 0
        self.wave_processed = False
        self.current_voice_state = 0
        self.voice_thread_running = True
        self.bg = ScrollingBackground("games/slogotakt/components/fon_bird3.png", speed=10)


        self.start_time = None
        self.airborne_time = 0
        self.is_airborne = False
        self.airborne_start_time = None

        self.voice_generator = None
        arcade.set_background_color(arcade.csscolor.SKY_BLUE)

    def setup(self):
        
        self.player = PlayerCharacter()

        self.gem_list.clear()
        self.wave_in_progress = False
        
        self.wave_collected = 0
        self.wave_failed = False
        self.score = 0
        self.max_possible_score = 0

        self.start_time = time.time()
        self.last_wave_time = self.start_time
        self.airborne_time = 0
        self.is_airborne = False
        self.airborne_start_time = None

        self.voice_generator = analyze_voice(self.game_duration, self.silence_threshold_db, self.blocks_to_silent)
        threading.Thread(target=self.voice_loop, daemon=True).start()
        self.start_new_wave()
    
    def voice_loop(self):
        for voice_state in self.voice_generator:
            if not self.voice_thread_running:
                break
            self.current_voice_state = voice_state
    def on_close(self):
        self.voice_thread_running = False
        super().on_close()

    def on_draw(self):
        arcade.start_render()
        self.bg.draw()

        for gem in self.gem_list:
            gem.draw()

        self.player.draw()

        # arcade.draw_text(f"Score: {self.score}",
        #                  10, SCREEN_HEIGHT - 30,
        #                  arcade.color.BLACK, 20)
        # arcade.draw_text(f"Time left: {int(GAME_DURATION - (time.time() - self.start_time))}",
        #                  SCREEN_WIDTH - 120, SCREEN_HEIGHT - 30,
        #                  arcade.color.BLACK, 20)

        # arcade.draw_text(f"Air time: {self.airborne_time:.1f}s",
        #                  SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30,
        #                  arcade.color.BLACK, 20, anchor_x="center")

        if time.time() - self.start_time > self.game_duration:
            arcade.draw_text("GAME OVER!",
                             SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2,
                             arcade.color.RED, 40, anchor_x="center")
            arcade.draw_text(f"Final Score: {self.score} / {self.max_possible_score}",
                             SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50,
                             arcade.color.WHITE, 30, anchor_x="center")

    def on_update(self, delta_time):
        if time.time() - self.start_time > self.game_duration:
            return

        self.bg.update()

        for gem in self.gem_list:
            gem.update()

        is_voiced = (self.current_voice_state == 1)
        self.player.update_physics(is_voiced)

        # Считаем таймер полета
        if self.player.center_y > GROUND_Y:
            if not self.is_airborne:
                self.is_airborne = True
                self.airborne_start_time = time.time()
            else:
                self.airborne_time = time.time() - self.airborne_start_time
        else:
            if self.is_airborne:
                self.is_airborne = False
                self.airborne_time = 0

        self.player.update_animation(delta_time)

        # Проверка столкновений
        for gem in self.gem_list:
            if gem.right < 10:
                gem.remove_from_sprite_lists()
                self.wave_failed = True

            if gem.check_collision_and_collect(self.player):
                if len(gem.letters) == 0:  # все буквы собраны
                    self.wave_collected += 1

        self.gem_list = [g for g in self.gem_list if len(g.letters) > 0]

        # Завершение волны
        if self.wave_in_progress and (self.wave_collected == self.wave_size or self.wave_failed):
            if not self.wave_failed:
                self.score += 10
            self.wave_in_progress = False
            self.last_wave_time = time.time()

        # Запуск новой волны через WAVE_INTERVAL
        if not self.wave_in_progress and (time.time() - self.last_wave_time >= WAVE_INTERVAL):
            self.start_new_wave()

    # ---------------------- САМАЯ ГЛАВНАЯ ЧАСТЬ: добавляем гемы по таймеру ----------------------
    def start_new_wave(self):
        """Запускаем волну и добавляем гемы один за другим через равные промежутки."""
        self.wave_in_progress = True
        self.wave_failed = False
        self.wave_collected = 0
        
        self.gem_count = 0
        self.max_possible_score += 10

        self.gem_list.clear()

        # Запускаем таймер, который будет вызывать add_gem() каждые COLLECTION_TIME / WAVE_SIZE сек.
        arcade.schedule(self.add_gem, self.syllable_interval)
        print("Началась новая волна!")

    def add_gem(self, delta_time=0):
        """Добавляет один гем (слово current_task) за каждый вызов schedule."""
        if self.gem_count < self.wave_size:
            gem_x = SCREEN_WIDTH + 100
            gem_y = AIR_Y
            new_gem = Gem(gem_x, gem_y, self.current_task)
            self.gem_list.append(new_gem)
            self.gem_count += 1
            print(f"Добавлен гем {self.gem_count}/{self.wave_size}")
        else:
            # Если все гемы добавлены, останавливаем таймер
            arcade.unschedule(self.add_gem)
    def run(self):
        arcade.run()
