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

class SEAKey(BaseModel):
    key: str

class SEAInput(BaseModel):
    text: str


SEA_ENG = None
tickets_processed = None


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

@app.post("/start")
def start(input_data: SEAKey):
    global SEA_ENG
    global tickets_processed

    SEA_ENG = OpenAIEmbeddings(
        api_key=input_data.key, model='text-embedding-3-large')

    tickets = connection()
    tickets_processed = []
    for ticket in tickets:
        concatenated_str = f"{ticket['title']} {
            ticket['content']} {ticket['description']}"

        tickets_processed.append(SEA_ENG.embed_query(concatenated_str))
    tickets_processed = np.vstack(tickets_processed)

    return {"success": "SEA_ENG initialized successfully"}
    

@app.post("/sea")
async def SEA(input_data: SEAInput):
    if SEA_ENG is None:
        return {"error": "SEA_ENG is not initialized"}

    input_text = input_data.text
    fc = SEA_ENG.embed_query(input_text)
    fc = np.array(fc).astype("float32")

    similarities = cosine_similarity(fc, tickets_processed)
    most_similar_index = np.argmax(similarities)
    tickets = connection()
    most_similar_ticket = tickets[most_similar_index]

    return most_similar_ticket['content']
