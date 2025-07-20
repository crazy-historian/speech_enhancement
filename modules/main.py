from game.core_game import *
import arcade

if __name__ == "__main__":
    game = VoiceBowlingGame()
    game.setup()
    arcade.run()