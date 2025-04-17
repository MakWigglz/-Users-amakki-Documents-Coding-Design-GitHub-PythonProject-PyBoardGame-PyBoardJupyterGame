import sqlite3
import tkinter as tk
from tkinter import simpledialog, messagebox
import random
class Square:
    board_size = 64
    board_color = ["black", "white", "red", "yellow", "green"]
    user_input = ["read", "unread", "like", "explain"]
    @classmethod
    def available_colors(cls):
        """Class method to return all available colors"""
        return cls.board_color
    @classmethod
    def user_feedback(cls):
        """Class method to return user feedback options"""
        return cls.user_input
    def __init__(self, row, column, title, paragraph, canvas_coords):
        self.row = row  # Object attribute
        self.column = column
        self.title = title
        self.paragraph = paragraph
        self.canvas_coords = canvas_coords  # Store coordinates for drawing
    def display_info(self):
        return f"Title: {self.title}\nParagraph: {self.paragraph}"
def load_square_data_from_db(db_path):
    """Load square data from the SQLite database."""
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        cursor.execute("SELECT title, content FROM posts")  # Adjust table and column names as needed
        rows = cursor.fetchall()
        connection.close()
        # Check if the number of rows is correct
        if len(rows) != Square.board_size:
            raise ValueError(
                f"Database must contain {Square.board_size} entries."
            )
        # Prepare the data in the required format
        return [{'title': row[0], 'paragraph': row[1]} for row in rows]
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None
    except ValueError as e:
        print(f"Error loading data: {e}")
        return None
class BoardGame:
    def __init__(self, master, db_path):
        self.master = master
        master.title("Educational Board Game")
        self.canvas_width = 600
        self.canvas_height = 600
        self.board_rows = 8
        self.board_cols = 8
        self.square_size = self.canvas_width // self.board_cols
        self.canvas = tk.Canvas(master, width=self.canvas_width, height=self.canvas_height, bg="red")
        self.canvas.pack()
        # Load data from the database
        self.square_data = load_square_data_from_db(db_path)
        if self.square_data:
            self.squares = self.create_board()
            self.draw_board()
            self.player_position = 0  # start at the first square
            self.player_piece = None
            self.create_player_piece()
            self.dice_button = tk.Button(master, text="Roll Dice", command=self.roll_dice)
            self.dice_button.pack(pady=10)
            self.info_label = tk.Label(master, text="")
            self.info_label.pack()
            self.display_square_info(self.player_position)
        else:
            # Handle the case where data loading failed
            messagebox.showerror(
                "Error",
                "Failed to load game data. The application will close."
            )
            master.destroy()
    def create_board(self):
        squares = []
        n = 0
        for r in range(self.board_rows):
            row_squares = []
            for c in range(self.board_cols):
                x1 = c * self.square_size
                y1 = r * self.square_size
                x2 = (c + 1) * self.square_size
                y2 = (r + 1) * self.square_size
                canvas_coords = ((x1 + x2) // 2, (y1 + y2) // 2)
                if self.square_data and n < len(self.square_data):
                    title = self.square_data[n]['title']
                    paragraph = self.square_data[n]['paragraph']
                    square = Square(r, c, title, paragraph, canvas_coords)
                    row_squares.append(square)
                    n += 1
                else:
                    square = Square(r, c, "Error", "Data not loaded correctly.", canvas_coords)
                    row_squares.append(square)
                    n += 1
            squares.append(row_squares)
        return [sq for row in squares for sq in row]  # Flatten the 2D list
    def draw_board(self):
        for i, square in enumerate(self.squares):
            row = i // self.board_cols
            col = i % self.board_cols
            x1 = col * self.square_size
            y1 = row * self.square_size
            x2 = (col + 1) * self.square_size
            y2 = (row + 1) * self.square_size
            # Set alternating colors for the board squares
            color = "white" if (row + col) % 2 == 0 else "lightyellow"
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")
            self.canvas.create_text(square.canvas_coords, text=str(i + 1), font=("Arial", 10, "bold"))
    def create_player_piece(self):
        start_square = self.squares[self.player_position]
        x, y = start_square.canvas_coords
        self.player_piece = self.canvas.create_oval(x - 10, y - 10, x + 10, y + 10, fill="red")
    def move_player(self, steps):
        current_square = self.squares[self.player_position]
        current_x, current_y = self.canvas.coords(self.player_piece)[0] + 10, self.canvas.coords(self.player_piece)[1] + 10  # Get center
        self.player_position = (self.player_position + steps) % Square.board_size
        next_square = self.squares[self.player_position]
        next_x, next_y = next_square.canvas_coords
        # Animate the movement
        dx = next_x - current_x
        dy = next_y - current_y
        frames = 30
        for i in range(frames + 1):
            self.canvas.coords(self.player_piece,
                               current_x - 10 + dx * i / frames,
                               current_y - 10 + dy * i / frames,
                               current_x + 10 + dx * i / frames,
                               current_y + 10 + dy * i / frames)
            self.master.update()
            self.master.after(20)
        self.display_square_info(self.player_position)
    def roll_dice(self):
        dice_roll = random.randint(1, 6)
        messagebox.showinfo("Dice Roll", f"You rolled a {dice_roll}!")
        self.move_player(dice_roll)
    def display_square_info(self, position):
        if 0 <= position < len(self.squares):
            square = self.squares[position]
            self.info_label.config(text=square.display_info())
        else:
            self.info_label.config(text="Error: Invalid square position.")
if __name__ == "__main__":
    root = tk.Tk()
    db_path = 'my_database.db'  # Path to your SQLite database
    game = BoardGame(root, db_path)
    root.mainloop()
