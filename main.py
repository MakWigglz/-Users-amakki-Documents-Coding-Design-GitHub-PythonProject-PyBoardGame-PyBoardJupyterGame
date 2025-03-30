# main.py
from database.database_setup import create_database
from database.queries import get_all_paragraphs, add_paragraph
DB_PATH = 'your_database_name.db'  # Define the database path
# Initialize the database
create_database(DB_PATH)
# Example usage of queries
add_paragraph(DB_PATH, "This is a sample paragraph.")
paragraphs = get_all_paragraphs(DB_PATH)
print(paragraphs)