import json
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///posts.db'  # Use SQLite for simplicity
db = SQLAlchemy(app)
class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text, nullable=False)
# Create the database
with app.app_context():
    db.create_all()
# Load JSON data and insert into the database
with open('data.json') as f:
    posts = json.load(f)
    for post in posts:
        new_post = Post(title=post['title'], content=post['content'])
        db.session.add(new_post)
    db.session.commit()