# database/queries.py
import sqlite3
def get_all_paragraphs(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM paragraphs")
    paragraphs = cursor.fetchall()
    conn.close()
    return paragraphs
def add_paragraph(db_path, paragraph):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO paragraphs (text) VALUES (?)", (paragraph,))
    conn.commit()
    conn.close()