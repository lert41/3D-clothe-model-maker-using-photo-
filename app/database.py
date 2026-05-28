import sqlite3

DB_NAME = "app.db"


def get_connection():
    return sqlite3.connect(DB_NAME)


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS clothing_sets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        folder_path TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        clothing_set_id INTEGER,
        image_path TEXT,
        image_type TEXT,
        FOREIGN KEY (clothing_set_id) REFERENCES clothing_sets(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS masks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        clothing_set_id INTEGER,
        mask_path TEXT,
        mask_type TEXT,
        FOREIGN KEY (clothing_set_id) REFERENCES clothing_sets(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS measurements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        clothing_set_id INTEGER,
        shoulder_width REAL,
        clothing_width REAL,
        clothing_height REAL,
        FOREIGN KEY (clothing_set_id) REFERENCES clothing_sets(id)
    )
    """)

    conn.commit()
    conn.close()
