from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import uvicorn

app = FastAPI()

# This allows your HTML file to talk to this Python script
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model you just saved
pipe = pickle.load(open('pipe.pkl', 'rb'))

class MatchInput(BaseModel):
    batting_team: str
    bowling_team: str
    city: str
    target: int
    score: int
    overs: float
    wickets: int

@app.post("/predict")
def predict(data: MatchInput):
    # Logic to transform user input into model features
    runs_left = data.target - data.score
    balls_left = 120 - (data.overs * 6)
    wickets_left = 10 - data.wickets
    crr = data.score / data.overs if data.overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    input_df = pd.DataFrame({
        'batting_team': [data.batting_team],
        'bowling_team': [data.bowling_team],
        'city': [data.city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets_left': [wickets_left],
        'target': [data.target],
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
    import os
    # Render provides a $PORT environment variable automatically
    port = int(os.environ.get("PORT", 8000)) 
    uvicorn.run(app, host="0.0.0.0", port=port)