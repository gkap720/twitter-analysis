from fastapi import FastAPI
import torch
import gensim.downloader as api
from ..data.process import clean_text, create_embedding
from ..models.sentiment_model import SentimentModel
import os

app = FastAPI()

model = SentimentModel(50, 20)
dirname = os.path.dirname(__file__)
model_path = os.path.join(dirname, "./models/best_model.pth")
model.load_state_dict(torch.load(model_path))
word2vec_transfer = api.load("glove-twitter-50")

@app.get("/")
def predict(input_str: str):
    tokenized = clean_text(input_str)
    vectorized = create_embedding(tokenized, word2vec_transfer)
    with torch.no_grad():
        model.eval()
        output = model(vectorized.unsqueeze(0))
        predicted_class = 1 if output.item() > 0.5 else 0
    return {"prediction": predicted_class}
