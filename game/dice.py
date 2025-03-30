# game/dice.py
import random
class Dice:
    def roll(self):
        return random.randint(1, 6)