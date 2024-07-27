import sqlite3
import pandas as pd


def create_table(db_path='sea.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sead (
            id INTEGER PRIMARY KEY,
            title TEXT,
            content TEXT,
            description TEXT
        )
    ''')
    conn.commit()
    conn.close()


create_table()


def insert_data(db_path='sea.db'):
    data = []
    df = pd.read_excel('sea.xlsx')

    def get_row(x):
        data.append((x['title'], x['content'], x['description']))

    df.apply(get_row, axis=1)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for entry in data:
        cursor.execute('''
            INSERT INTO sead (title, content, description) VALUES (?, ?, ?)
        ''', entry)

    conn.commit()
    conn.close()


create_table()

insert_data()
