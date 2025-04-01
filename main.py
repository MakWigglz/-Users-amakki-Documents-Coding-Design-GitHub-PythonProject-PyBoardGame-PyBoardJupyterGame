from flask import Flask, render_template, request, redirect, url_for
from database.database_setup import create_database
from database.queries import get_all_paragraphs, add_paragraph
from board import Board

app = Flask(__name__)
DB_PATH = '/Users/amakki/Documents/Coding-Design/GitHub/PythonProject/PyBoardGame/PyBoardJupyterGame/game_data.db'

# Initialize the database
create_database(DB_PATH)

@app.route('/')
def home():
    # Fetch all paragraphs to display on the homepage
    paragraphs = get_all_paragraphs(DB_PATH)
    return render_template('index.html', paragraphs=paragraphs)

@app.route('/add_paragraph', methods=['POST'])
def add_paragraph_route():
    # Get the paragraph text from the form
    paragraph_text = request.form.get('paragraph')
    if paragraph_text:
        add_paragraph(DB_PATH, paragraph_text)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)