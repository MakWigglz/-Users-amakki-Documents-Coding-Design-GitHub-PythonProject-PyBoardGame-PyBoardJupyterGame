import sqlite3
import json
import os

def setup_database(db_path, json_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS topics (
            topic TEXT PRIMARY KEY,
            paragraphs TEXT
        )
    ''')
    with open(json_path, 'r') as f:
        data = json.load(f)
        for item in data:
            cursor.executemany('''
    INSERT OR REPLACE INTO subjects (title, content)
    VALUES (?, ?);
''', [(entry['title'], entry['content']) for entry in data])
            
    conn.commit()
    conn.close()
if __name__ == "__main__":
    db_path = '/Users/amakki/Documents/Coding-Design/GitHub/PythonProject/PyBoardGame/PyBoardJupyterGame/game_data.db'
    json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'paragraphs.json')
    setup_database(db_path, json_path)
            
