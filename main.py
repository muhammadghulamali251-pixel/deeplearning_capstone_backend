import sys
sys.path.append('..')
import uvicorn
import joblib
from tensorflow.keras.models import load_model
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np

from text_processing import encode, pad_id

# App
app = FastAPI()

# Cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173",
                   "https://review-ai-seven.vercel.app"
                   ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuickReview(BaseModel):
    review: str

# Deep learning model(LSTM)
model = load_model('models/lstm.keras')
vocab = joblib.load('models/vocab.pkl')

@app.post('/quick/predict')
def predictPractice(text:QuickReview):
    try:
        encoded = encode(pd.Series([text.review]), vocab)
        padded = pad_id(encoded)
        prob = model.predict(padded, verbose=0).tolist()[0][0]
        sentiment = "Positive" if prob > 0.65 else "Uncertain" if prob > 0.35 else "Negative"
        prob = 1 - prob if sentiment == "Negative" else prob
        return {"Review": text.review, "Sentiment": sentiment, "Probability": prob}

    except HTTPException:
        raise    
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail="An error occurred.")


@app.post('/batch/predict')
def predict_review(file: UploadFile):
    try:
        # Validation
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

        df = pd.read_csv(file.file)

        if len(df.columns) > 1:
            raise HTTPException(status_code=400, detail="Invalid file. Upload a CSV with a single column containing the review text.")

        if "review" not in df.columns.str.lower():
            raise HTTPException(status_code=400, detail="Invalid file. The CSV must contain a column named 'review'.")

        if df.shape[0] > 1000:
            raise HTTPException (status_code=400, detail=f"Limit exceeded. The CSV must not have more than 1000 rows." )    

        # Preprocessing
        encoded = encode(df['Review'], vocab)
        padded = pad_id(encoded)

        # Predict
        prediction = model.predict(padded, verbose=0).flatten()

        df['Sentiment'] = np.select(
            [prediction > 0.65, prediction < 0.35],
            ['Positive', 'Negative'],
            default='Uncertain'
        )

        # Return
        df['Probability'] = np.round(np.where(prediction < 0.35, 1 - prediction, prediction), 3)


        return (df.to_dict(orient='records'))
    
    except HTTPException:
        raise

    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")        

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)