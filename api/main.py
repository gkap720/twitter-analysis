from fastapi import FastAPI
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences
import gensim.downloader as api
from utils import process_text, create_embedding
import numpy as np
import tensorflow as tf

app = FastAPI()

new_model = load_model('models/twitter_lstm.keras')
word2vec_transfer = api.load("glove-twitter-50")

@app.get("/")
def predict(input_str: str):
    tokenized = process_text(input_str)
    vectorized = np.expand_dims(create_embedding(tokenized, word2vec_transfer), axis=0)
    padded = pad_sequences(vectorized, padding="post", dtype="float32", maxlen=35)
    with tf.device('/cpu:0'):
        prediction = new_model.predict(padded)
    return {"prediction": float(prediction[0, 0])}
