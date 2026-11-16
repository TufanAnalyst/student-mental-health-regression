from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI(title="Student Mental Health Score API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("mental_health_pipeline.joblib")

TOP_COUNTRIES = [
    "Other", "India", "USA", "Canada", "Australia",
    "UK", "Germany", "Mexico", "France", "Turkey",
]


class UserInput(BaseModel):
    Study_Hours: float = Field(ge=0, le=9)
    Age: int = Field(ge=18, le=24)
    Avg_Daily_Usage_Hours: float = Field(ge=0, le=12)
    Daily_Unlocks: int = Field(ge=0, le=300)
    Physical_Activity_Hours: float = Field(ge=0, le=5)
    Sleep_Hours_Per_Night: float = Field(ge=3, le=10)
    Stress_Level: Literal["Low", "Medium", "High", "Very High"]
    Gender: Literal["Male", "Female"]
    Academic_Level: Literal["High School", "Undergraduate", "Graduate"]
    Most_Used_Platform: Literal[
        "Facebook", "Instagram", "KakaoTalk", "LINE", "LinkedIn", "Snapchat",
        "TikTok", "Twitter", "VKontakte", "WeChat", "WhatsApp", "YouTube",
    ]
    Purpose_Of_Use: Literal["Education", "Entertainment", "Networking", "News"]
    Country: str = Field(min_length=1)


class PredictionOut(BaseModel):
    predicted_mental_health_score: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
def predict(data: UserInput):
    row = data.model_dump()

    if row["Country"] not in TOP_COUNTRIES:
        row["Country"] = "Other"

    df = pd.DataFrame([row])
    prediction = float(model.predict(df)[0])

    return {"predicted_mental_health_score": round(prediction, 2)}
