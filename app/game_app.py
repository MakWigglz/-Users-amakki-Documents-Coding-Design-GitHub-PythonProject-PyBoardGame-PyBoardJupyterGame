from flask import Flask, render_template, g, redirect, url_for
import sqlite3
import random
import os
import json

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates'))

DATABASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'game_data.db')

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

@app.route('/')
def board():
    if 'player' not in g:
        g.player = 1
    if 'position' not in g:
        g.position = 0
    if 'dice_roll' not in g:
        g.dice_roll = 0
    if 'paragraphs' not in g:
        g.paragraphs = ""
    return render_template('board.html', player=g.player, position=g.position, dice_roll=g.dice_roll, paragraphs = g.paragraphs)

@app.route('/roll_dice')
def roll_dice():
    g.dice_roll = random.randint(1, 8)
    g.position = (g.position + g.dice_roll) % 64
    g.player = 2 if g.player == 1 else 1
    return redirect(url_for('board'))

@app.route('/topic/<int:topic_num>')
def topic(topic_num):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT paragraphs FROM topics WHERE topic = ?", (f"Topic {topic_num}",))
    result = cursor.fetchone()
    if result:
        g.paragraphs = json.loads(result[0])
    else:
        g.paragraphs = "Topic not found."
    return redirect(url_for('board'))

@app.route('/exit')
def exit_game():
    g.pop('player', None)
    g.pop('position', None)
    g.pop('dice_roll', None)
    g.pop('paragraphs', None)
    return redirect(url_for('board'))

if __name__ == '__main__':
    app.run(debug=True)
