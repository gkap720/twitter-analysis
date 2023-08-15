import re
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.nn.utils.rnn import pad_sequence
import os
import argparse
import math

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
    return torch.tensor(np.array(output), dtype=torch.float32)

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    target = data[0].map({"0": 0, "4": 1})
    tweets = data[5]
    cleaned = tweets.map(clean_text)
    return pd.DataFrame({"target": target, "tweet": cleaned})

def collate_fn(batch):
    filtered_batch = [item for item in batch if item is not None]
    if len(filtered_batch) == 0:
        return None
    texts, labels, lengths = zip(*filtered_batch)
    padded_texts = pad_sequence(texts, batch_first=True)
    return padded_texts, torch.tensor(labels), torch.tensor(lengths)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("processor")
    parser.add_argument("-size", "--training-size", 
                        help="The size of the train set. Test and val sets are 30% of this size", 
                        type=int, default=50000)
    args = parser.parse_args()
    
    train_size = args.training_size #200_000
    val_size = math.floor(train_size * 0.3) #60_000
    test_size = math.floor(train_size * 0.3) # 60_000
    dirname = os.path.dirname(__file__)
    data_path = os.path.join(dirname, "../../data/training.1600000.processed.noemoticon.csv")

    # the data is sorted first by negative reviews and then positive
    negative = pd.read_csv(data_path, 
        header=None, encoding="ISO-8859-1", 
        dtype=str, nrows=(train_size+val_size+test_size)/2)
    positive = pd.read_csv(data_path, 
        header=None, encoding="ISO-8859-1", nrows=(train_size+val_size+test_size)/2,
        dtype=str, skiprows=800_000)
    
    # only perform string cleaning and tokenization, embedding creates very large files
    neg_proc = process_data(negative)
    pos_proc = process_data(positive)

    # split up into train, test and val for easy use with dataloader later on
    neg_train, neg_inter = train_test_split(neg_proc, test_size=test_size)
    pos_train, pos_inter = train_test_split(pos_proc, test_size=test_size)
    df_train = pd.concat([neg_train, pos_train])
    neg_test, neg_val = train_test_split(neg_inter, test_size=0.5)
    pos_test, pos_val = train_test_split(pos_inter, test_size=0.5)
    df_test = pd.concat([neg_test, pos_test])
    df_val = pd.concat([neg_val, pos_val])
    df_train.to_csv(os.path.join(dirname, "../../data/train.csv"))
    df_test.to_csv(os.path.join(dirname, "../../data/test.csv"))
    df_val.to_csv(os.path.join(dirname, "../../data/val.csv"))
    
