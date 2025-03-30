# game/token.py
class Token:
    def __init__(self, player_name):
        self.player_name = player_name
        self.position = 0
    def move(self, steps):
        self.position += steps