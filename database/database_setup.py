# database/database_setup.py
import sqlite3
def create_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create a table for paragraphs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS paragraphs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            paragraphs TEXT
        );
    ''')
    
    conn.commit()
    conn.close()
if __name__ == "__main__":
    create_database('/Users/amakki/Documents/Coding-Design/GitHub/PythonProject/PyBoardGame/PyBoardJupyterGame/game_data.db')  # Replace with your actual database path