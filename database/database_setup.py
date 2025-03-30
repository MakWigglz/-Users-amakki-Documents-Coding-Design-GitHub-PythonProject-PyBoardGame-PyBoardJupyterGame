# database/database_setup.py
import sqlite3
def create_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create a table for paragraphs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS paragraphs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()
if __name__ == "__main__":
    create_database('your_database_name.db')  # Replace with your actual database path