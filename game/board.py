# game/board.py
class Board:
    def __init__(self):
        self.spaces = self.create_spaces()
    def create_spaces(self):
        # Create spaces on the board
        return ["Start", "Space 1", "Space 2", "Space 3", "Finish"]
    def display(self):
        # Logic to display the board
        print("Board Spaces:", self.spaces)