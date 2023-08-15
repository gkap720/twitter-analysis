import re
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
import numpy as np

stops = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def process_text(sentence: str):
    # remove urls
    sentence = re.sub(r'https?://\S+', "", sentence)
    # remove special chars and numbers
    sentence = re.sub(r'[^A-Za-z ]', "", sentence)
    sentence = sentence.lower()
    tokenized = word_tokenize(sentence)
    lemmatized = [lemmatizer.lemmatize(word) for word in tokenized]
    lemmatized = [lemmatizer.lemmatize(word, pos="v") for word in lemmatized]
    return [word for word in lemmatized if word not in stops]

def create_embedding(tokenized: list, w2v):
    output = []
    for word in tokenized:
        if word in w2v:
            output.append(w2v.get_vector(word, norm=True))
    return np.array(output)