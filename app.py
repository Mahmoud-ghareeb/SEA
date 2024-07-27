import sqlite3
import numpy as np
from langchain_openai import OpenAIEmbeddings
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import os


load_dotenv()

app = FastAPI()

SEA_ENG = OpenAIEmbeddings(
    api_key=os.getenv('OPENAI_API_KEY'), model='text-embedding-3-large')


class SEAInput(BaseModel):
    text: str


def connection(db_path='sea.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = "SELECT title, content, description FROM sead"
    cursor.execute(query)

    data = cursor.fetchall()
    conn.close()

    tickets = []
    for row in data:
        tickets.append({
            "title": row[0],
            "content": row[1],
            "description": row[2]
        })

    return tickets


def cosine_similarity(a, b):
    return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1))


def embedding_engine():
    tickets = connection()
    tickets_processed = []
    for ticket in tickets:
        concatenated_str = f"{ticket['title']} {ticket['content']} {ticket['description']}"
        tickets_processed.append(SEA_ENG.embed_query(concatenated_str))
    tickets_processed = np.vstack(tickets_processed)
    return tickets_processed


tickets_processed = embedding_engine()


@app.post("/sea")
async def SEA(input_data: SEAInput):
    input_text = input_data.text
    fc = SEA_ENG.embed_query(input_text)
    fc = np.array(fc).astype("float32")

    similarities = cosine_similarity(fc, tickets_processed)
    most_similar_index = np.argmax(similarities)
    tickets = connection()
    most_similar_ticket = tickets[most_similar_index]

    return most_similar_ticket['content']
