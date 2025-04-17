from flask import Flask
from board_graphical_rep import generate_graph  # Assuming this function exists
app = Flask(__name__)
@app.route('/graph')
def graph():
    generate_graph()  # Call the function to generate the graph
    return "Graph generated!"
if __name__ == '__main__':
    app.run()
