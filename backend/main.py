from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import uvicorn
import os

app = FastAPI()

# Enable CORS so Vercel can talk to Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Robust model loading
MODEL_PATH = 'pipe.pkl'
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = 'backend/pipe.pkl'

with open(MODEL_PATH, 'rb') as f:
    pipe = pickle.load(f)

class MatchInput(BaseModel):
    batting_team: str
    bowling_team: str
    city: str
    target: int
    score: int
    overs: float
    wickets: int

@app.get("/")
def home():
    return {"status": "API is running. Use /predict for predictions."}

@app.post("/predict")
def predict(data: MatchInput):
    # Logic to transform user input into model features
    runs_left = data.target - data.score
    balls_left = 120 - (data.overs * 6)
    wickets_left = 10 - data.wickets
    
    # Calculate rates
    crr = data.score / data.overs if data.overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    # CRITICAL: These column names MUST match your training dataframe
    input_df = pd.DataFrame({
        'batting_team': [data.batting_team],
        'bowling_team': [data.bowling_team],
        'city': [data.city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_left],      # Renamed from 'wickets_left'
        'total_runs_x': [data.target],  # Renamed from 'target'
        'crr': [crr],
        'rrr': [rrr]
    })

    # Predict probability
    result = pipe.predict_proba(input_df)
    
    return {
        "batting_win": round(result[0][1] * 100),
        "bowling_win": round(result[0][0] * 100)
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000)) 
    uvicorn.run(app, host="0.0.0.0", port=port)