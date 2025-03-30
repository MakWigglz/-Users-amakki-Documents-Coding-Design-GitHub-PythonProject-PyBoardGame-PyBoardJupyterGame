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
    VALUES ('Geology', 'Geology is the study of the Earth's physical structure, substances, and processes. It investigates how tectonic plates shape mountains and valleys. Volcanology explores magma, eruptions, and lava formations. Seismology studies earthquakes and seismic waves. Mineralogy examines the chemical composition of Earth's minerals. Paleontology, a subfield, analyzes fossils to uncover past life forms. Geomorphology explains how landscapes evolve over time. Hydrogeology looks at groundwater flow and its impact. Stratigraphy examines layered rock to interpret Earth's historical changes. Geologists study resources like oil, gas, and metals. Environmental geology assesses how human activities affect our planet, promoting sustainability and disaster mitigation.'),
    ('Chemistry', 'Chemistry is the study of the properties and behavior of matter. It includes valent and covalent bonds of the electron shells or chemical compositions that make up complex biological and physical structures. At the fundamental level we can talk about quantum physics as the basic composition of matter.'),
    ('Mathematics', 'Mathematics is the abstract science of numbers, quantity, and space. Arithmetic forms the basis, focusing on numbers and operations. Algebra introduces variables and equations. Geometry explores shapes, sizes, and properties of space. Trigonometry studies angles and their relationships. Calculus deals with change and motion, using derivatives and integrals. Probability and statistics analyze uncertainty and data patterns. Discrete mathematics examines finite systems. Topology investigates spatial properties that remain unchanged through deformation. Applied mathematics solves real-world problems in physics, engineering, and economics. Mathematics is vital for technological advancements, fostering logic and innovation in numerous domains.')
    ;
''', [(entry['title'], entry['content']) for entry in data])
            
    conn.commit()
    conn.close()
if __name__ == "__main__":
    db_path = '/Users/amakki/Documents/Coding-Design/GitHub/PythonProject/PyBoardGame/PyBoardJupyterGame/game_data.db'
    json_path = '/Users/amakki/Documents/Coding-Design/GitHub/PythonProject/PyBoardGame/PyBoardJupyterGame/data/paragraphs.json'
    setup_database(db_path, json_path)
            
