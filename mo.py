import uvicorn
from fastapi import FastAPI, HTTPException
import numpy as np
import pickle
import pandas as pd
import joblib
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer
from fastapi.middleware.cors import CORSMiddleware

# 2. Create the app object
app = FastAPI()
model = joblib.load("sihmodel.pkl")
count_vectorizer = joblib.load("count_vectorizer.pkl")


@app.get('/')
def index():
    return {'message': 'Hello, World sih model'}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to allow requests only from your frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/recommend/")
async def recommend_service(user_input: dict):
    input_text = user_input.get("user_input")
    input_data= count_vectorizer.transform([input_text])
    
    try:
        recommendation = model.predict(input_data)
        return {"recommendation": recommendation[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#uvicorn mo:app --reload