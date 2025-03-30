from flask import Flask, render_template, jsonify
from game import game  # Import your game logic
from database import queries  # Import your database queries
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/data')
def get_data():
    # Load data from paragraphs.json
    with open('data/paragraphs.json') as f:
        data = json.load(f)
    return jsonify(data)
if __name__ == '__main__':
    app.run(debug=True)