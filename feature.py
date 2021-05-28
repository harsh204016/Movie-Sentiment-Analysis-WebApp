import re
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

stop_words = pd.read_csv("data/stopwords.csv")

def text_cleaning(text):
    forbidden_words = stop_words['text'].to_list()
    if text:
        text = ' '.join(text.split('.'))
        text = re.sub('\/',' ',text)
        text = re.sub(r'\\',' ',text)
        text = re.sub(r'((http)\S+)','',text)
        text = re.sub(r'\s+', ' ', re.sub('[^A-Za-z]', ' ', text.strip().lower())).strip()
        text = re.sub(r'\W+', ' ', text.strip().lower()).strip()
        text = [word for word in text.split() if word not in forbidden_words]
        return text
    return []

def text_encoding(text):
    vocab_size = 50000
    maxlen = 200
    text = ' '.join(text)
    h = one_hot(text,vocab_size)
    l=[]
    l.append(h)
    x = pad_sequences(l, maxlen=maxlen)
    return x
    

