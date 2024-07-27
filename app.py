import sqlite3
import numpy as np
from langchain_openai import OpenAIEmbeddings
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from chromadb.client import PersistentClient
from chromadb import Chroma

app = FastAPI()
SEA_ENG = OpenAIEmbeddings(api_key='sk-proj-fMPLXVUCyOWPCFeuM4NcT3BlbkFJbABO62e51cxt3pGvrGuu', model='text-embedding-3-large')

class SEAInput(BaseModel):
    input: str

class SEAInput(BaseModel):
    input: str

def cosine_similarity(a, b):
    return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1))

@app.post("/embedding_engine")
def embedding_engine(db_path='sea.db', name='ticket_collection', embeddings=SEA_ENG):
    
    tickets = connection(db_path)
    client = PersistentClient(path='store')

    try:
        collection = client.get_collection(name=name)
        print('Collection already exists')
    except Exception as e:
        print(f'Collection does not exist, creating a new one: {e}')
        collection = client.get_or_create_collection(name=name)

    tickets_processed = []
    for ticket in tickets:
        concatenated_str = f"{ticket['title']} {ticket['content']} {ticket['description']}"
        embedded_vector = embeddings(concatenated_str)
        tickets_processed.append({
            'document': concatenated_str,
            'vector': embedded_vector
        })

    for ticket in tickets_processed:
        collection.add(document=ticket['document'], vector=ticket['vector'])

    return collection

vector_db = embedding_engine()

@app.post("/sea")
async def SEA(input_data: SEAInput):
    input_text = input_data.input
    fc = SEA_ENG.embed_query(input_text)
    fc = np.array(fc).astype("float32")

    similarities = cosine_similarity(fc, tickets_processed)
    most_similar_index = np.argmax(similarities)
    tickets = connection()
    most_similar_ticket = tickets[most_similar_index]

    return {"ticket": most_similar_ticket['content']}
