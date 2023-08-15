import re
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pandas as pd
from sklearn.model_selection import train_test_split
import gensim.downloader as api

stops = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(sentence: str):
    # remove urls
    sentence = re.sub(r'https?://\S+', "", sentence)
    # remove user tags
    sentence = re.sub(r'@\S+', "", sentence)
    # remove special chars and numbers
    sentence = re.sub(r'[^A-Za-z ]', "", sentence)
    sentence = sentence.lower()
    # tokenize sentence (IE split into words)
    tokenized = word_tokenize(sentence)
    # use lemmatizer to reduce words to roots when possible (tried -> try)
    lemmatized = [lemmatizer.lemmatize(word) for word in tokenized]
    lemmatized = [lemmatizer.lemmatize(word, pos="v") for word in lemmatized]
    # remove stop words before returning
    return [word for word in lemmatized if word not in stops]

def create_embedding(tokenized: list, w2v):
    output = []
    for word in tokenized:
        # if word exists in embedding, find its associated vector
        if word in w2v:
            output.append(w2v.get_vector(word, norm=True))
    return np.array(output)

def pad_sequences(sequences):
    # Sort sequences by length
    sorted_sequences = sorted(sequences, key=lambda x: len(x), reverse=True)

    # Convert to a packed sequence
    packed_sequences = pack_padded_sequence(sorted_sequences, enforce_sorted=False)

    # Pad the packed sequence
    padded_packed_sequences, _ = pad_packed_sequence(packed_sequences, batch_first=True)
    return padded_packed_sequences

def process_data(data: pd.DataFrame, w2v) -> pd.DataFrame:
    target = data[0].map({"0": 0, "4": 1})
    tweets = data[5]
    cleaned = tweets.map(clean_text)
    embedded = cleaned.map(lambda x: create_embedding(x, w2v))
    return pd.DataFrame({"target": target, "tweet": embedded})

if __name__ == "__main__":
    train_size = 200_000 #200_000
    val_size = 60_000 #60_000
    test_size = 60_000 # 60_000
    data_path = "../../data/training.1600000.processed.noemoticon.csv"
    word2vec_transfer = api.load("glove-twitter-50")

    # the data is sorted first by negative reviews and then positive
    negative = pd.read_csv(data_path, 
        header=None, encoding="ISO-8859-1", 
        dtype=str, nrows=(train_size+val_size+test_size)/2)
    positive = pd.read_csv(data_path, 
        header=None, encoding="ISO-8859-1", nrows=(train_size+val_size+test_size)/2,
        dtype=str, skiprows=800_000)
    neg_proc = process_data(negative, word2vec_transfer)
    pos_proc = process_data(positive, word2vec_transfer)
    neg_train, neg_inter = train_test_split(neg_proc, test_size=test_size)
    pos_train, pos_inter = train_test_split(pos_proc, test_size=test_size)
    df_train = pd.concat([neg_train, pos_train])
    neg_test, neg_val = train_test_split(neg_inter, test_size=0.5)
    pos_test, pos_val = train_test_split(pos_inter, test_size=0.5)
    df_test = pd.concat([neg_test, pos_test])
    df_val = pd.concat([neg_val, pos_val])
    df_train.to_csv("../../data/train.csv")
    df_test.to_csv("../../data/test.csv")
    df_val.to_csv("../../data/val.csv")
    
