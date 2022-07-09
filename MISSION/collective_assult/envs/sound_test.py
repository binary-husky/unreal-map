# Source: https://techwithtim.net/tutorials/game-development-with-python/pygame-tutorial/sound-effects-music/
# and: https://realpython.com/playing-and-recording-sound-python/

from pygame import mixer  # Load the popular external library
import time

mixer.init()
mixer.music.load('Game/bullet.mp3')
 
for i in range(5):
	mixer.music.play()
	time.sleep(0.5)